"""Forecasting backbones and the TS-IFA adapter."""

from .chronos_model import Chronos
from .models import ForecastModel, load_model, load_pretrained_model, resolve_device
from .patchtst import PatchTST
from .tabpfn_model import TabPFNTS
from ..adaptors.ts_ifa.model import TSIFAConfig, TimeSeriesInformedForecastingAdapter

__all__ = [
    "Chronos",
    "ChronosBolt",
    "ForecastModel",
    "PatchTST",
    "TabPFNTS",
    "TSIFAConfig",
    "TimeSeriesInformedForecastingAdapter",
    "load_model",
    "load_pretrained_model",
    "resolve_device",
]


def __getattr__(name: str):
    if name == "ChronosBolt":
        from .chronos_bolt import ChronosBolt

        return ChronosBolt
    raise AttributeError(name)
