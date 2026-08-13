"""Chronos-T5 retriever used by the released TS-RAG implementation."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class TSRAGRetriever(nn.Module):
    """Return the EOS embedding from ``amazon/chronos-t5-base``."""

    def __init__(
        self,
        weights_path: str | Path,
        *,
        device_map: str = "cuda",
        local_files_only: bool = True,
    ) -> None:
        super().__init__()
        from chronos import BaseChronosPipeline

        path = Path(weights_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        self.pipeline = BaseChronosPipeline.from_pretrained(
            str(path),
            device_map=device_map,
            local_files_only=local_files_only,
        )
        model = getattr(self.pipeline, "model", None)
        if model is not None:
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad = False

    @torch.no_grad()
    def representation(
        self,
        x: torch.Tensor,
        *,
        pool: bool = False,
    ) -> torch.Tensor:
        del pool
        embeddings, _ = self.pipeline.embed(x.squeeze(1))
        return embeddings[:, -1, :].float().to(x.device)
