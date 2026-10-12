"""
config.py — load and expose seekaudio configuration as dataclasses.

Usage:
    from seekaudio.config import load_config
    cfg = load_config()                        # uses config/default.yaml
    cfg = load_config("config/default.yaml", overrides={"retrieval": {"k": 5}})
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import yaml


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AudioConfig:
    sample_rate: int = 22050
    duration: int = 30
    mono: bool = True


@dataclass
class FeatureGroupsConfig:
    mfcc: bool = True
    spectral: bool = True
    rhythm: bool = True
    energy: bool = True
    tonal: bool = False


@dataclass
class FeatureWeightsConfig:
    mfcc: float = 1.0
    spectral: float = 1.0
    rhythm: float = 1.0
    energy: float = 1.0
    tonal: float = 1.0


@dataclass
class FeaturesConfig:
    n_mfcc: int = 20
    n_fft: int = 2048
    hop_length: int = 512
    n_chroma: int = 12
    n_mels: int = 128
    groups: FeatureGroupsConfig = field(default_factory=FeatureGroupsConfig)
    weights: FeatureWeightsConfig = field(default_factory=FeatureWeightsConfig)


@dataclass
class PoolingConfig:
    use_std: bool = True


@dataclass
class SpaceConfig:
    pca_components: Optional[int] = 40
    whiten: bool = True
    l2_normalize: bool = True


@dataclass
class RetrievalConfig:
    k: int = 10


@dataclass
class PathsConfig:
    audio_dir: str = "data/raw"
    manifest: str = "data/manifest.parquet"
    features_dir: str = "data/features"
    feature_spec_version: int = 2
    space_dir: str = "data/space"
    evaluation_dir: str = "data/evaluation"


@dataclass
class SeekAudioConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    space: SpaceConfig = field(default_factory=SpaceConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    pooling: PoolingConfig = field(default_factory=PoolingConfig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (non-destructive)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _dict_to_config(d: dict) -> SeekAudioConfig:
    d = copy.deepcopy(d)
    audio = AudioConfig(**d.get("audio", {}))

    feat_d = d.get("features", {})
    groups_d = feat_d.pop("groups", {})
    weights_d = feat_d.pop("weights", {})
    features = FeaturesConfig(
        **feat_d,
        groups=FeatureGroupsConfig(**groups_d),
        weights=FeatureWeightsConfig(**weights_d),
    )

    space = SpaceConfig(**d.get("space", {}))
    retrieval = RetrievalConfig(**d.get("retrieval", {}))
    paths = PathsConfig(**d.get("paths", {}))

    return SeekAudioConfig(
        audio=audio,
        features=features,
        space=space,
        retrieval=retrieval,
        paths=paths,
        pooling=PoolingConfig(**d.get("pooling", {})),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(
    path: str | Path = "config/default.yaml",
    overrides: Optional[Dict] = None,
) -> SeekAudioConfig:
    """Load config from *path* YAML, optionally deep-merging *overrides*."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open() as f:
        raw = yaml.safe_load(f) or {}

    if overrides:
        raw = _deep_merge(raw, overrides)

    return _dict_to_config(raw)


def load_ablation_configs(
    default_path: str | Path = "config/default.yaml",
    ablation_path: str | Path = "config/ablation.yaml",
) -> Dict[str, SeekAudioConfig]:
    """Return {name: SeekAudioConfig} for every entry in ablation.yaml."""
    default_path = Path(default_path)
    ablation_path = Path(ablation_path)

    with default_path.open() as f:
        base = yaml.safe_load(f) or {}

    with ablation_path.open() as f:
        ablation = yaml.safe_load(f) or {}

    configs: Dict[str, SeekAudioConfig] = {}
    for name, override in ablation.get("configs", {}).items():
        merged = _deep_merge(base, override or {})
        configs[name] = _dict_to_config(merged)

    return configs
