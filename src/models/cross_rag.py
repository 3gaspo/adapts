"""Released Cross-RAG model adapted to the project's tensor pipeline.

The architecture is copied from https://github.com/seunghan96/cross-rag at
commit b9a5428365b8ada43a986b2501ece12dd3844e95. Training, data loading,
retrieval, and general-purpose pipeline classes are omitted. The model layers
and released ``moe`` forward computation are retained; only explicit
constructor inputs and transformers 5 compatibility were added.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from packaging import version
from transformers import __version__ as transformers_version
from transformers.models.t5.modeling_t5 import (
    ACT2FN,
    T5Config,
    T5LayerNorm,
    T5PreTrainedModel,
    T5Stack,
)
from transformers.utils import ModelOutput


_TRANSFORMERS_V5 = version.parse(transformers_version) >= version.parse("5.0.0")
if _TRANSFORMERS_V5:
    from transformers import initialization as init
else:
    from torch.nn import init


def _create_t5_stack(config: T5Config, embed_tokens: nn.Embedding) -> T5Stack:
    if _TRANSFORMERS_V5:
        return T5Stack(config)
    return T5Stack(config, embed_tokens)


@dataclass
class ChronosBoltConfig:
    context_length: int
    prediction_length: int
    input_patch_size: int
    input_patch_stride: int
    quantiles: List[float]
    use_reg_token: bool = False


@dataclass
class ChronosBoltOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    quantile_preds: Optional[torch.Tensor] = None
    attentions: Optional[torch.Tensor] = None
    cross_attentions: Optional[torch.Tensor] = None


class Patch(nn.Module):
    def __init__(self, patch_size: int, patch_stride: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        if length % self.patch_size != 0:
            padding_size = (
                *x.shape[:-1],
                self.patch_size - (length % self.patch_size),
            )
            padding = torch.full(
                size=padding_size,
                fill_value=torch.nan,
                dtype=x.dtype,
                device=x.device,
            )
            x = torch.concat((padding, x), dim=-1)
        return x.unfold(
            dimension=-1,
            size=self.patch_size,
            step=self.patch_stride,
        )


class InstanceNorm(nn.Module):
    """Cross-RAG's released constant-aware instance normalization."""

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        loc_scale: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if loc_scale is None:
            loc = torch.nan_to_num(
                torch.nanmean(x, dim=-1, keepdim=True),
                nan=0.0,
            )
            scale = torch.nan_to_num(
                (x - loc).square().nanmean(dim=-1, keepdim=True).sqrt(),
                nan=1.0,
            )
            is_constant = torch.all(x == x[..., :1], dim=-1, keepdim=True)
            scale = torch.where(is_constant, torch.ones_like(scale), scale)
        else:
            loc, scale = loc_scale
        normalized = (x - loc) / scale
        is_constant = (
            torch.all(x == x[..., :1], dim=-1, keepdim=True)
            if loc_scale is None
            else scale == 1
        )
        normalized = torch.where(
            is_constant,
            torch.ones_like(normalized),
            normalized,
        )
        return normalized, (loc, scale)

    def inverse(
        self,
        x: torch.Tensor,
        loc_scale: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        loc, scale = loc_scale
        return torch.where(scale == 1, loc, x * scale + loc)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        act_fn_name: str,
        dropout_p: float = 0.0,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.hidden_layer = nn.Linear(in_dim, h_dim)
        self.act = ACT2FN[act_fn_name]
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = T5LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.act(self.hidden_layer(x))
        output = self.dropout(self.output_layer(hidden))
        output = output + self.residual_layer(x)
        if self.use_layer_norm:
            return self.layer_norm(output)
        return output


class ChronosBoltModelForForecastingWithRetrieval(T5PreTrainedModel):
    """Cross-RAG's Chronos-Bolt backbone with dual retrieval-attention heads."""

    _keys_to_ignore_on_load_missing = [
        r"input_patch_embedding\.",
        r"output_patch_embedding\.",
    ]
    _keys_to_ignore_on_load_unexpected = [r"lm_head.weight"]
    _tied_weights_keys = (
        {
            "encoder.embed_tokens.weight": "shared.weight",
            "decoder.embed_tokens.weight": "shared.weight",
        }
        if _TRANSFORMERS_V5
        else ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]
    )

    def __init__(
        self,
        config: T5Config,
        augment: str = "moe",
        context_length: int = 512,
        mix_lambda: float = 0.7,
    ) -> None:
        assert hasattr(config, "chronos_config"), "Not a Chronos config file"
        super().__init__(config)
        self.model_dim = config.d_model
        self.augment = augment
        config_fields = {field.name for field in fields(ChronosBoltConfig)}
        self.chronos_config = ChronosBoltConfig(
            **{
                key: value
                for key, value in config.chronos_config.items()
                if key in config_fields
            }
        )
        if self.chronos_config.use_reg_token:
            config.reg_token_id = 1
        config.vocab_size = 2 if self.chronos_config.use_reg_token else 1
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.input_patch_embedding = ResidualBlock(
            in_dim=self.chronos_config.input_patch_size * 2,
            h_dim=config.d_ff,
            out_dim=config.d_model,
            act_fn_name=config.dense_act_fn,
            dropout_p=config.dropout_rate,
        )
        self.patch = Patch(
            patch_size=self.chronos_config.input_patch_size,
            patch_stride=self.chronos_config.input_patch_stride,
        )
        self.instance_norm = InstanceNorm()
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = _create_t5_stack(encoder_config, self.shared)
        self._init_decoder(config)
        self.mix_lambda = float(mix_lambda)
        self.context_length = int(context_length)
        self.num_quantiles = len(self.chronos_config.quantiles)
        quantiles = torch.tensor(self.chronos_config.quantiles, dtype=self.dtype)
        self.quantiles: torch.Tensor
        self.register_buffer("quantiles", quantiles, persistent=False)
        self.output_patch_embedding = ResidualBlock(
            in_dim=config.d_model,
            h_dim=config.d_ff,
            out_dim=self.num_quantiles * self.chronos_config.prediction_length,
            act_fn_name=config.dense_act_fn,
            dropout_p=config.dropout_rate,
        )
        self.dropout = nn.Dropout(p=0.2)
        self.encode_mlp_x = nn.Sequential(
            nn.Linear(self.context_length, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.encode_mlp_y = nn.Sequential(
            nn.Linear(self.chronos_config.prediction_length, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.cross_mha = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=8,
            batch_first=True,
        )
        self.self_mha = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=8,
            batch_first=True,
        )
        self.ffn_cross = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.ffn_self = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.post_init()
        self.model_parallel = False
        self.device_map = None

    def _init_weights(self, module):
        super()._init_weights(module)
        factor = self.config.initializer_factor
        if isinstance(module, self.__class__):
            init.normal_(module.shared.weight, mean=0.0, std=factor)
            if _TRANSFORMERS_V5:
                quantiles = torch.tensor(
                    module.chronos_config.quantiles,
                    dtype=module.dtype,
                    device=module.quantiles.device,
                )
                init.copy_(module.quantiles, quantiles)
        elif isinstance(module, ResidualBlock):
            patch_scale = (self.chronos_config.input_patch_size * 2) ** -0.5
            init.normal_(
                module.hidden_layer.weight,
                mean=0.0,
                std=factor * patch_scale,
            )
            if module.hidden_layer.bias is not None:
                init.zeros_(module.hidden_layer.bias)
            init.normal_(
                module.residual_layer.weight,
                mean=0.0,
                std=factor * patch_scale,
            )
            if module.residual_layer.bias is not None:
                init.zeros_(module.residual_layer.bias)
            init.normal_(
                module.output_layer.weight,
                mean=0.0,
                std=factor * self.config.d_ff**-0.5,
            )
            if module.output_layer.bias is not None:
                init.zeros_(module.output_layer.bias)

    def forward(
        self,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        retrieved_seq: Optional[torch.Tensor] = None,
        distances: Optional[torch.Tensor] = None,
    ) -> ChronosBoltOutput:
        mask = (
            mask.to(context.dtype)
            if mask is not None
            else torch.isnan(context).logical_not().to(context.dtype)
        )
        batch_size, _ = context.shape
        if context.shape[-1] > self.chronos_config.context_length:
            context = context[..., -self.chronos_config.context_length :]
            mask = mask[..., -self.chronos_config.context_length :]
        context, loc_scale = self.instance_norm(context)
        retrieved_seq, _ = self.instance_norm(retrieved_seq)
        if "moe" not in self.augment:
            weights = torch.softmax(-distances, dim=1)
            retrieved_seq = (weights.unsqueeze(-1) * retrieved_seq).sum(dim=1)
            retrieved_seq = retrieved_seq.unsqueeze(1)
        retrieved_batch, retrieved_count, retrieved_length = retrieved_seq.shape
        assert retrieved_batch == batch_size
        forecast_length = self.chronos_config.prediction_length
        retrieved_x, retrieved_y = retrieved_seq.split(
            (retrieved_length - forecast_length, forecast_length),
            dim=2,
        )
        context = context.to(self.dtype)
        mask = mask.to(self.dtype)
        retrieved_seq = retrieved_seq.to(self.dtype)
        patched_context = self.patch(context)
        patched_mask = torch.nan_to_num(self.patch(mask), nan=0.0)
        patched_context = torch.where(
            patched_mask > 0,
            patched_context,
            0.0,
        )
        patched_context = torch.cat([patched_context, patched_mask], dim=-1)
        attention_mask = patched_mask.sum(dim=-1) > 0
        input_embeds = self.input_patch_embedding(patched_context)
        if self.chronos_config.use_reg_token:
            reg_input_ids = torch.full(
                (batch_size, 1),
                self.config.reg_token_id,
                device=input_embeds.device,
            )
            reg_embeds = self.shared(reg_input_ids)
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(reg_input_ids)],
                dim=-1,
            )
        encoder_outputs = self.encoder(
            attention_mask=attention_mask,
            inputs_embeds=input_embeds,
        )
        hidden_states = encoder_outputs[0]
        sequence_output = self.decode(
            input_embeds,
            attention_mask,
            hidden_states,
        )
        if self.augment == "moe":
            retrieved_x_enc = []
            retrieved_y_enc = []
            for index in range(retrieved_count):
                retrieved_x_enc.append(self.encode_mlp_x(retrieved_x[:, index, :]))
                retrieved_y_enc.append(self.encode_mlp_y(retrieved_y[:, index, :]))
            retrieved_x_enc = torch.stack(retrieved_x_enc, dim=1)
            retrieved_y_enc = torch.stack(retrieved_y_enc, dim=1)
            cross_out, _ = self.cross_mha(
                sequence_output,
                retrieved_x_enc,
                retrieved_y_enc,
            )
            cross_out = sequence_output + cross_out
            cross_out = cross_out + self.dropout(self.ffn_cross(cross_out))
            self_out, _ = self.self_mha(
                retrieved_y_enc,
                retrieved_y_enc,
                retrieved_y_enc,
            )
            self_out = retrieved_y_enc + self_out
            self_out = self_out + self.dropout(self.ffn_self(self_out))
            self_pool = self_out.mean(dim=1, keepdim=True)
            sequence_output = (
                self.mix_lambda * cross_out
                + (1 - self.mix_lambda) * self_pool
            )
        prediction_shape = (
            batch_size,
            self.num_quantiles,
            self.chronos_config.prediction_length,
        )
        quantile_preds = self.output_patch_embedding(sequence_output).view(
            *prediction_shape
        )
        loss = None
        if target is not None:
            target, _ = self.instance_norm(target, loc_scale)
            target = target.unsqueeze(1)
            assert self.chronos_config.prediction_length >= target.shape[-1]
            target = target.to(quantile_preds.device)
            target_mask = (
                target_mask.unsqueeze(1).to(quantile_preds.device)
                if target_mask is not None
                else ~torch.isnan(target)
            )
            target[~target_mask] = 0.0
            if self.chronos_config.prediction_length > target.shape[-1]:
                padding_shape = (
                    *target.shape[:-1],
                    self.chronos_config.prediction_length - target.shape[-1],
                )
                target = torch.cat(
                    [target, torch.zeros(padding_shape).to(target)],
                    dim=-1,
                )
                target_mask = torch.cat(
                    [target_mask, torch.zeros(padding_shape).to(target_mask)],
                    dim=-1,
                )
            loss = (
                2
                * torch.abs(
                    (target - quantile_preds)
                    * (
                        (target <= quantile_preds).float()
                        - self.quantiles.view(1, self.num_quantiles, 1)
                    )
                )
                * target_mask.float()
            )
            loss = loss.mean(dim=-2).sum(dim=-1).mean()
        quantile_preds = self.instance_norm.inverse(
            quantile_preds.view(batch_size, -1),
            loc_scale,
        ).view(*prediction_shape)
        return ChronosBoltOutput(loss=loss, quantile_preds=quantile_preds)

    def _init_decoder(self, config: T5Config) -> None:
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = _create_t5_stack(decoder_config, self.shared)

    def decode(
        self,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        decoder_input_ids = torch.full(
            (input_embeds.shape[0], 1),
            self.config.decoder_start_token_id,
            device=input_embeds.device,
        )
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            output_attentions=output_attentions,
            return_dict=True,
        )
        return decoder_outputs.last_hidden_state
