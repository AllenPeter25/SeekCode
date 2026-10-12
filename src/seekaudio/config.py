import copy
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class AudioConfig:
    sample_rate: int = 22_050
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
    n_fft: int = 2_048
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
    pca_components: int | None = 40
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


def read_yaml(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def _deep_merge(base: dict, overrides: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _dict_to_config(settings: dict) -> SeekAudioConfig:
    feature_settings = copy.deepcopy(settings.get("features", {}))
    group_settings = feature_settings.pop("groups", {})
    weight_settings = feature_settings.pop("weights", {})
    features = FeaturesConfig(
        **feature_settings,
        groups=FeatureGroupsConfig(**group_settings),
        weights=FeatureWeightsConfig(**weight_settings),
    )
    return SeekAudioConfig(
        audio=AudioConfig(**settings.get("audio", {})),
        features=features,
        space=SpaceConfig(**settings.get("space", {})),
        retrieval=RetrievalConfig(**settings.get("retrieval", {})),
        paths=PathsConfig(**settings.get("paths", {})),
        pooling=PoolingConfig(**settings.get("pooling", {})),
    )


def load_config(path: str | Path = "config/default.yaml", overrides: dict | None = None) -> SeekAudioConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    settings = read_yaml(path)
    if overrides:
        settings = _deep_merge(settings, overrides)
    return _dict_to_config(settings)


def load_ablation_configs(
    default_path: str | Path = "config/default.yaml",
    ablation_path: str | Path = "config/ablation.yaml",
) -> dict[str, SeekAudioConfig]:
    default_settings = read_yaml(default_path)
    experiment_settings = read_yaml(ablation_path).get("configs", {})
    configs = {}
    for name, overrides in experiment_settings.items():
        settings = _deep_merge(default_settings, overrides or {})
        configs[name] = _dict_to_config(settings)
    return configs
