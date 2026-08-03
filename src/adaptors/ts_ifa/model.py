"""TS-IFA: Time Series Informed Forecasting Adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
from einops import pack, rearrange


CANDIDATE_NAMES = ("vanilla", "cov", "transformed", "residual", "memory")
BASE_CANDIDATE_NAMES = ("vanilla", "cov", "residual", "memory")


def mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


def zero_initialize_output(module: nn.Sequential) -> None:
    output = module[-1]
    if not isinstance(output, nn.Linear):
        raise TypeError("expected the last module to be a Linear layer")
    nn.init.zeros_(output.weight)
    nn.init.zeros_(output.bias)


def initialize_vanilla_skip(module: nn.Linear, horizon: int) -> None:
    """Initialize ``module([p, z])`` to exactly return ``p``."""
    if module.in_features != 2 * horizon or module.out_features != horizon:
        raise ValueError("vanilla skip must map 2H inputs to H outputs")
    nn.init.zeros_(module.weight)
    nn.init.zeros_(module.bias)
    with torch.no_grad():
        module.weight[:, :horizon].copy_(torch.eye(horizon))


class CrossAttentionBlock(nn.Module):
    """Projected multi-head cross-attention followed by norm and feed-forward layers."""

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        value_dim: int,
        output_dim: int,
        *,
        heads: int = 4,
        attn_dim: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.heads = int(heads)
        self.attn_dim = int(attn_dim)
        self.model_dim = self.heads * self.attn_dim
        self.query_projection = nn.Linear(query_dim, self.model_dim)
        self.key_projection = nn.Linear(key_dim, self.model_dim)
        self.value_projection = nn.Linear(value_dim, self.model_dim)
        self.query_norm = nn.LayerNorm(self.model_dim)
        self.key_norm = nn.LayerNorm(self.model_dim)
        self.value_norm = nn.LayerNorm(self.model_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.model_dim,
            num_heads=self.heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(self.model_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.model_dim, 4 * self.model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * self.model_dim, self.model_dim),
            nn.Dropout(dropout),
        )
        self.feed_forward_norm = nn.LayerNorm(self.model_dim)
        self.output_projection = nn.Linear(self.model_dim, output_dim)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = rearrange(
            self.query_norm(self.query_projection(query)),
            "batch dim -> batch 1 dim",
        )
        k = self.key_norm(self.key_projection(keys))
        v = self.value_norm(self.value_projection(values))
        attended, weights = self.attention(
            q,
            k,
            v,
            need_weights=True,
            average_attn_weights=True,
        )
        hidden = self.attention_norm(q + attended)
        hidden = self.feed_forward_norm(hidden + self.feed_forward(hidden))
        output = self.output_projection(rearrange(hidden, "batch 1 dim -> batch dim"))
        weights = rearrange(weights, "batch 1 items -> batch items")
        return output, weights


@dataclass(frozen=True)
class TSIFAConfig:
    lags: int
    horizon: int
    neighbors: int
    residual_heads: int = 4
    memory_heads: int = 4
    rooter_heads: int = 4
    residual_attn_dim: int = 32
    memory_attn_dim: int = 32
    rooter_attn_dim: int = 32
    residual_hidden: int = 128
    memory_hidden: int = 128
    rooter_hidden: int = 128
    transformed_hidden: int = 128
    precomputed_transformed_expert: bool = False
    learnable_transformed_covariate: bool = False
    vanilla_anchoring_init: bool = True
    dropout: float = 0.0

    @property
    def rooter_dim(self) -> int:
        return int(self.rooter_heads) * int(self.rooter_attn_dim)

    @property
    def has_transformed_expert(self) -> bool:
        return bool(
            self.precomputed_transformed_expert
            or self.learnable_transformed_covariate
        )


class TimeSeriesInformedForecastingAdapter(nn.Module):
    """Four-expert TS-IFA with one optional transformed candidate."""

    def __init__(self, config: TSIFAConfig):
        super().__init__()
        if config.neighbors <= 0:
            raise ValueError("TimeSeriesInformedForecastingAdapter requires neighbors > 0")
        self.config = config
        lags = int(config.lags)
        horizon = int(config.horizon)
        rooter_dim = int(config.rooter_dim)
        self.candidate_names = (
            CANDIDATE_NAMES if config.has_transformed_expert else BASE_CANDIDATE_NAMES
        )

        # The residual branch retrieves errors from analogous states [history, prediction].
        self.residual_attention = CrossAttentionBlock(
            query_dim=lags + horizon,
            key_dim=lags + horizon,
            value_dim=horizon,
            output_dim=horizon,
            heads=config.residual_heads,
            attn_dim=config.residual_attn_dim,
            dropout=config.dropout,
        )
        self.residual_head = mlp(
            2 * horizon,
            config.residual_hidden,
            horizon,
            dropout=config.dropout,
        )

        # The memory branch maps [vanilla forecast, retrieved-horizon state] directly
        # to a forecast. Its explicit skip permits an exact vanilla initialization.
        self.memory_attention = CrossAttentionBlock(
            query_dim=lags,
            key_dim=lags,
            value_dim=horizon,
            output_dim=horizon,
            heads=config.memory_heads,
            attn_dim=config.memory_attn_dim,
            dropout=config.dropout,
        )
        self.memory_skip = nn.Linear(2 * horizon, horizon)
        self.memory_head = mlp(
            2 * horizon,
            config.memory_hidden,
            horizon,
            dropout=config.dropout,
        )

        if config.learnable_transformed_covariate:
            self.transformed_covariate = mlp(
                lags,
                config.transformed_hidden,
                horizon,
                dropout=config.dropout,
            )
            self.transformed_head = mlp(
                2 * horizon,
                config.transformed_hidden,
                horizon,
                dropout=config.dropout,
            )
        else:
            self.transformed_covariate = None
            self.transformed_head = None

        # The rooter learns its own retrieval representation from x and Xc; it does
        # not receive precomputed distances or retrieval-dispersion features.
        self.rooter_attention = CrossAttentionBlock(
            query_dim=lags,
            key_dim=lags,
            value_dim=lags,
            output_dim=rooter_dim,
            heads=config.rooter_heads,
            attn_dim=config.rooter_attn_dim,
            dropout=config.dropout,
        )
        self.rooter_global_norm = nn.LayerNorm(rooter_dim)
        self.rooter_forecast_norm = nn.LayerNorm(horizon)
        self.candidate_tokens = nn.Embedding(len(self.candidate_names), rooter_dim)
        self.candidate_token_norm = nn.LayerNorm(rooter_dim)
        self.rooter_scorer = mlp(
            2 * rooter_dim + 2 * horizon,
            config.rooter_hidden,
            horizon,
            dropout=config.dropout,
        )
        nn.init.normal_(self.candidate_tokens.weight, mean=0.0, std=0.02)

        if config.vanilla_anchoring_init:
            zero_initialize_output(self.residual_head)
            initialize_vanilla_skip(self.memory_skip, horizon)
            zero_initialize_output(self.memory_head)
            if self.transformed_head is not None:
                zero_initialize_output(self.transformed_head)
            zero_initialize_output(self.rooter_scorer)

    def branch_modules(self) -> tuple[nn.Module, ...]:
        modules: list[nn.Module] = [
            self.residual_attention,
            self.residual_head,
            self.memory_attention,
            self.memory_skip,
            self.memory_head,
        ]
        if self.transformed_covariate is not None and self.transformed_head is not None:
            modules.extend([self.transformed_covariate, self.transformed_head])
        return tuple(modules)

    def rooter_modules(self) -> tuple[nn.Module, ...]:
        return (
            self.rooter_attention,
            self.rooter_global_norm,
            self.rooter_forecast_norm,
            self.candidate_tokens,
            self.candidate_token_norm,
            self.rooter_scorer,
        )

    @staticmethod
    def _module_parameters(modules: Iterable[nn.Module]) -> list[nn.Parameter]:
        parameters: list[nn.Parameter] = []
        seen: set[int] = set()
        for module in modules:
            for parameter in module.parameters():
                if id(parameter) not in seen:
                    seen.add(id(parameter))
                    parameters.append(parameter)
        return parameters

    def branch_parameters(self) -> list[nn.Parameter]:
        return self._module_parameters(self.branch_modules())

    def rooter_parameters(self) -> list[nn.Parameter]:
        return self._module_parameters(self.rooter_modules())

    def set_trainable_stage(self, stage: str) -> None:
        """Enable only branch, rooter, or all parameters for optimization."""
        selected: set[int]
        if stage == "branches":
            selected = {id(parameter) for parameter in self.branch_parameters()}
        elif stage == "rooter":
            selected = {id(parameter) for parameter in self.rooter_parameters()}
        elif stage == "all":
            selected = {id(parameter) for parameter in self.parameters()}
        else:
            raise ValueError(f"unknown TS-IFA stage {stage!r}")
        for parameter in self.parameters():
            parameter.requires_grad = id(parameter) in selected

    def forward_branches(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Build the active candidates without executing the neural rooter."""
        x = batch["x"]
        x_c = batch["x_c"]
        y_c = batch["y_c"]
        pred = batch["pred"]
        pred_cov = batch["pred_cov"]
        pred_transformed = batch["pred_transformed"]
        pred_neighbors = batch["pred_neighbors"]
        residual_c = batch["residual_c"]

        query_state, _ = pack([x, pred], "batch *")
        neighbor_states, _ = pack([x_c, pred_neighbors], "batch neighbor *")
        z_r, residual_weights = self.residual_attention(
            query_state,
            neighbor_states,
            residual_c,
        )
        residual_input, _ = pack([pred, z_r], "batch *")
        residual_delta = self.residual_head(residual_input)
        residual_prediction = pred + residual_delta

        z_m, memory_weights = self.memory_attention(x, x_c, y_c)
        memory_input, _ = pack([pred, z_m], "batch *")
        memory_prediction = self.memory_skip(memory_input) + self.memory_head(memory_input)

        transformed_prediction = pred_transformed
        transformed_delta = None
        learned_covariate = None
        if self.transformed_covariate is not None and self.transformed_head is not None:
            learned_covariate = self.transformed_covariate(x)
            transformed_input, _ = pack(
                [pred_transformed, learned_covariate],
                "batch *",
            )
            transformed_delta = self.transformed_head(transformed_input)
            transformed_prediction = pred_transformed + transformed_delta

        candidate_values = [pred, pred_cov]
        if self.config.has_transformed_expert:
            candidate_values.append(transformed_prediction)
        candidate_values.extend([residual_prediction, memory_prediction])
        candidates = torch.stack(candidate_values, dim=1)
        outputs = {
            "candidates": candidates,
            "residual_prediction": residual_prediction,
            "memory_prediction": memory_prediction,
            "residual_delta": residual_delta,
            "residual_weights": residual_weights,
            "memory_weights": memory_weights,
        }
        if self.config.has_transformed_expert:
            outputs["transformed_prediction"] = transformed_prediction
        if transformed_delta is not None:
            outputs["transformed_delta"] = transformed_delta
            outputs["transformed_covariate"] = learned_covariate
        return outputs

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = batch["x"]
        x_c = batch["x_c"]
        pred = batch["pred"]
        branch_outputs = self.forward_branches(batch)
        candidates = branch_outputs["candidates"]
        z_g, rooter_retrieval_weights = self.rooter_attention(x, x_c, x_c)
        batch_size, candidate_count, _ = candidates.shape
        global_state = self.rooter_global_norm(z_g).unsqueeze(1).expand(
            batch_size,
            candidate_count,
            -1,
        )
        vanilla_state = self.rooter_forecast_norm(pred).unsqueeze(1).expand(
            batch_size,
            candidate_count,
            -1,
        )
        candidate_state = self.rooter_forecast_norm(candidates)
        token_ids = torch.arange(candidate_count, device=x.device)
        type_state = self.candidate_token_norm(self.candidate_tokens(token_ids))
        type_state = type_state.unsqueeze(0).expand(batch_size, -1, -1)
        scorer_input = torch.cat(
            [global_state, vanilla_state, candidate_state, type_state],
            dim=-1,
        )
        coefficients = self.rooter_scorer(
            rearrange(scorer_input, "batch candidate dim -> (batch candidate) dim")
        )
        coefficients = rearrange(
            coefficients,
            "(batch candidate) horizon -> batch candidate horizon",
            batch=batch_size,
            candidate=candidate_count,
        )
        prediction = pred + (coefficients * candidates).sum(dim=1)

        return {
            **branch_outputs,
            "prediction": prediction,
            "coefficients": coefficients,
            "rooter_state": z_g,
            "rooter_retrieval_weights": rooter_retrieval_weights,
        }


ProposedModelConfig = TSIFAConfig
