"""TS-IFA: Time Series Informed Forecasting Adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
from einops import pack, rearrange


BRANCH_NAMES = ("cov", "residual", "memory")
TSIFA_VARIANTS = ("joint_ridge", "joint_neural", "meta_ridge", "meta_neural")
ROUTING_SCOPES = ("shared", "horizon")
ROUTING_CONSTRAINTS = ("unconstrained", "softmax")
TSIFA_ARCHITECTURE = "configurable_delta_branches_routing_v4"


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


def initialize_output_bias(module: nn.Sequential, bias: float) -> None:
    output = module[-1]
    if not isinstance(output, nn.Linear):
        raise TypeError("expected the last module to be a Linear layer")
    nn.init.zeros_(output.weight)
    nn.init.constant_(output.bias, float(bias))


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
    rooter_form: str = "ridge"
    branches: tuple[str, ...] = BRANCH_NAMES
    routing_scope: str = "horizon"
    routing_constraint: str = "unconstrained"
    residual_heads: int = 4
    memory_heads: int = 4
    rooter_heads: int = 4
    residual_attn_dim: int = 32
    memory_attn_dim: int = 32
    rooter_attn_dim: int = 32
    residual_hidden: int = 128
    memory_hidden: int = 128
    rooter_hidden: int = 128
    vanilla_anchoring_init: bool = True
    dropout: float = 0.0

    @property
    def rooter_dim(self) -> int:
        return int(self.rooter_heads) * int(self.rooter_attn_dim)

class TimeSeriesInformedForecastingAdapter(nn.Module):
    """Configurable TS-IFA branches with shared or horizon routing."""

    def __init__(self, config: TSIFAConfig):
        super().__init__()
        if config.neighbors <= 0:
            raise ValueError("TimeSeriesInformedForecastingAdapter requires neighbors > 0")
        self.config = config
        if config.rooter_form not in {"ridge", "neural"}:
            raise ValueError(f"unknown TS-IFA rooter form {config.rooter_form!r}")
        if config.routing_scope not in ROUTING_SCOPES:
            raise ValueError(f"unknown TS-IFA routing scope {config.routing_scope!r}")
        if config.routing_constraint not in ROUTING_CONSTRAINTS:
            raise ValueError(
                f"unknown TS-IFA routing constraint {config.routing_constraint!r}"
            )
        branches = tuple(config.branches)
        if not branches or len(set(branches)) != len(branches):
            raise ValueError("TS-IFA branches must be a non-empty unique selection")
        unknown = set(branches) - set(BRANCH_NAMES)
        if unknown:
            raise ValueError(f"unknown TS-IFA branches: {sorted(unknown)}")
        lags = int(config.lags)
        horizon = int(config.horizon)
        rooter_dim = int(config.rooter_dim)
        self.active_branches = tuple(name for name in BRANCH_NAMES if name in branches)
        self.candidate_names = ("vanilla", *self.active_branches)
        self.rooter_candidate_names = (
            self.candidate_names
            if config.routing_constraint == "softmax"
            else self.active_branches
        )
        routing_output_dim = 1 if config.routing_scope == "shared" else horizon

        # The residual branch retrieves errors from analogous states [history, prediction].
        if "residual" in self.active_branches:
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

        # The memory branch predicts a correction to the vanilla forecast.
        if "memory" in self.active_branches:
            self.memory_attention = CrossAttentionBlock(
                query_dim=lags,
                key_dim=lags,
                value_dim=horizon,
                output_dim=horizon,
                heads=config.memory_heads,
                attn_dim=config.memory_attn_dim,
                dropout=config.dropout,
            )
            self.memory_head = mlp(
                2 * horizon,
                config.memory_hidden,
                horizon,
                dropout=config.dropout,
            )

        if config.rooter_form == "ridge":
            initial_value = (
                -8.0
                if config.routing_constraint == "softmax" and config.vanilla_anchoring_init
                else 0.0
            )
            self.ridge_coefficients = nn.Parameter(
                torch.full((len(self.active_branches), routing_output_dim), initial_value)
            )
        else:
            # The neural rooter learns its own retrieval representation from x and Xc.
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
            self.rooter_delta_norm = nn.LayerNorm(horizon)
            self.candidate_tokens = nn.Embedding(len(self.active_branches), rooter_dim)
            self.candidate_token_norm = nn.LayerNorm(rooter_dim)
            self.rooter_scorer = mlp(
                2 * rooter_dim + 2 * horizon,
                config.rooter_hidden,
                routing_output_dim,
                dropout=config.dropout,
            )
            nn.init.normal_(self.candidate_tokens.weight, mean=0.0, std=0.02)

        if config.vanilla_anchoring_init:
            if "residual" in self.active_branches:
                zero_initialize_output(self.residual_head)
            if "memory" in self.active_branches:
                zero_initialize_output(self.memory_head)
            if config.rooter_form == "neural":
                if config.routing_constraint == "softmax":
                    initialize_output_bias(self.rooter_scorer, -8.0)
                else:
                    zero_initialize_output(self.rooter_scorer)

    def branch_modules(self) -> tuple[nn.Module, ...]:
        modules: list[nn.Module] = []
        if "residual" in self.active_branches:
            modules.extend((self.residual_attention, self.residual_head))
        if "memory" in self.active_branches:
            modules.extend((self.memory_attention, self.memory_head))
        return tuple(modules)

    def rooter_modules(self) -> tuple[nn.Module, ...]:
        if self.config.rooter_form == "ridge":
            return ()
        return (
            self.rooter_attention,
            self.rooter_global_norm,
            self.rooter_forecast_norm,
            self.rooter_delta_norm,
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
        if self.config.rooter_form == "ridge":
            return [self.ridge_coefficients]
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
        pred_neighbors = batch["pred_neighbors"]
        residual_c = batch["residual_c"]

        candidate_values: dict[str, torch.Tensor] = {"vanilla": pred}
        outputs: dict[str, torch.Tensor] = {}
        if "cov" in self.active_branches:
            candidate_values["cov"] = pred_cov
        if "residual" in self.active_branches:
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
            candidate_values["residual"] = residual_prediction
            outputs.update(
                residual_prediction=residual_prediction,
                residual_delta=residual_delta,
                residual_weights=residual_weights,
            )
        if "memory" in self.active_branches:
            z_m, memory_weights = self.memory_attention(x, x_c, y_c)
            memory_input, _ = pack([pred, z_m], "batch *")
            memory_delta = self.memory_head(memory_input)
            memory_prediction = pred + memory_delta
            candidate_values["memory"] = memory_prediction
            outputs.update(
                memory_prediction=memory_prediction,
                memory_delta=memory_delta,
                memory_weights=memory_weights,
            )
        outputs["candidates"] = torch.stack(
            [candidate_values[name] for name in self.candidate_names], dim=1
        )
        return outputs

    def _expand_routing(self, value: torch.Tensor) -> torch.Tensor:
        if self.config.routing_scope == "shared":
            return value.expand(*value.shape[:-1], self.config.horizon)
        return value

    def _route(
        self,
        candidates: torch.Tensor,
        branch_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        branch_values = self._expand_routing(branch_values)
        if self.config.routing_constraint == "softmax":
            vanilla_logits = branch_values.new_zeros(
                branch_values.shape[0], 1, branch_values.shape[-1]
            )
            coefficients = torch.softmax(
                torch.cat((vanilla_logits, branch_values), dim=1), dim=1
            )
            prediction = (coefficients * candidates).sum(dim=1)
        else:
            coefficients = branch_values
            candidate_deltas = candidates[:, 1:] - candidates[:, :1]
            prediction = candidates[:, 0] + (coefficients * candidate_deltas).sum(dim=1)
        return prediction, coefficients

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = batch["x"]
        x_c = batch["x_c"]
        pred = batch["pred"]
        branch_outputs = self.forward_branches(batch)
        candidates = branch_outputs["candidates"]
        candidate_deltas = candidates[:, 1:] - pred.unsqueeze(1)
        batch_size, branch_count, _ = candidate_deltas.shape
        if self.config.rooter_form == "ridge":
            routing_values = self.ridge_coefficients.unsqueeze(0).expand(batch_size, -1, -1)
            prediction, coefficients = self._route(candidates, routing_values)
            return {
                **branch_outputs,
                "prediction": prediction,
                "coefficients": coefficients,
                "candidate_deltas": candidate_deltas,
                "routing_values": self._expand_routing(routing_values),
            }

        z_g, rooter_retrieval_weights = self.rooter_attention(x, x_c, x_c)
        global_state = self.rooter_global_norm(z_g).unsqueeze(1).expand(batch_size, branch_count, -1)
        vanilla_state = self.rooter_forecast_norm(pred).unsqueeze(1).expand(
            batch_size,
            branch_count,
            -1,
        )
        delta_state = self.rooter_delta_norm(candidate_deltas)
        token_ids = torch.arange(branch_count, device=x.device)
        type_state = self.candidate_token_norm(self.candidate_tokens(token_ids))
        type_state = type_state.unsqueeze(0).expand(batch_size, -1, -1)
        scorer_input = torch.cat(
            [global_state, vanilla_state, delta_state, type_state],
            dim=-1,
        )
        routing_values = self.rooter_scorer(
            rearrange(scorer_input, "batch candidate dim -> (batch candidate) dim")
        )
        routing_values = rearrange(
            routing_values,
            "(batch candidate) route -> batch candidate route",
            batch=batch_size,
            candidate=branch_count,
        )
        prediction, coefficients = self._route(candidates, routing_values)

        return {
            **branch_outputs,
            "prediction": prediction,
            "coefficients": coefficients,
            "candidate_deltas": candidate_deltas,
            "routing_values": self._expand_routing(routing_values),
            "rooter_state": z_g,
            "rooter_retrieval_weights": rooter_retrieval_weights,
        }


ProposedModelConfig = TSIFAConfig
