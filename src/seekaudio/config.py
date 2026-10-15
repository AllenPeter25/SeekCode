"""Turn YAML settings into named dataclasses used by the pipeline and CLI.

Defaults provide a runnable setup. An override replaces only specified values;
ablation.yaml supplies multiple named overrides of the same default settings.
Configuration loading does not extract audio or fit any models.
"""

import copy
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class AudioConfig:
    """Waveform preparation settings: samples/second, excerpt seconds, mono mode."""
    sample_rate: int = 22_050
    duration: int = 30
    mono: bool = True


@dataclass
class FeatureGroupsConfig:
    """Which groups are selected when building a search representation."""
    mfcc: bool = True
    spectral: bool = True
    rhythm: bool = True
    energy: bool = True
    tonal: bool = False


@dataclass
class FeatureWeightsConfig:
    """Multipliers applied after each selected group has been normalized."""
    mfcc: float = 1.0
    spectral: float = 1.0
    rhythm: float = 1.0
    energy: float = 1.0
    tonal: float = 1.0


@dataclass
class FeaturesConfig:
    """Extractor window sizes and coefficient counts, plus group selection/weights."""
    n_mfcc: int = 20
    n_fft: int = 2_048
    hop_length: int = 512
    n_chroma: int = 12
    n_mels: int = 128
    # A factory gives each config its own nested object, rather than shared mutable state.
    groups: FeatureGroupsConfig = field(default_factory=FeatureGroupsConfig)
    weights: FeatureWeightsConfig = field(default_factory=FeatureWeightsConfig)


@dataclass
class PoolingConfig:
    """Whether search-space selection keeps standard-deviation feature columns."""
    use_std: bool = True


@dataclass
class SpaceConfig:
    """Representation settings; pca_components=None skips dimensionality reduction."""
    pca_components: int | None = 40
    whiten: bool = True
    l2_normalize: bool = True


@dataclass
class RetrievalConfig:
    """Default number of returned matches or evaluated neighbors."""
    k: int = 10


@dataclass
class PathsConfig:
    """Dataset/artifact paths and a feature version separating extraction recipes."""
    audio_dir: str = "data/raw"
    manifest: str = "data/manifest.parquet"
    features_dir: str = "data/features"
    feature_spec_version: int = 2
    space_dir: str = "data/space"
    evaluation_dir: str = "data/evaluation"


@dataclass
class SeekAudioConfig:
    """Top-level configuration grouping all settings under readable attributes."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    space: SpaceConfig = field(default_factory=SpaceConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    pooling: PoolingConfig = field(default_factory=PoolingConfig)


def read_yaml(path: str | Path) -> dict:
    """Read UTF-8 YAML safely, treating an empty document as an empty dictionary."""
    with Path(path).open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge nested overrides without mutating the base dictionary.

    For example, overriding retrieval.k retains all unrelated audio/space paths.
    A dictionary merges recursively; a scalar or list replaces the old value.
    """
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _dict_to_config(settings: dict) -> SeekAudioConfig:
    """Convert nested YAML dictionaries into dataclasses with defaulted fields.

    Remove groups/weights from a copied feature dictionary before passing the
    remaining scalar settings to FeaturesConfig. This preserves the caller's data.
    """
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
    """Load one YAML file, optionally merge overrides, and return typed settings."""
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
    """Return a separate complete configuration for each named ablation override."""
    default_settings = read_yaml(default_path)
    experiment_settings = read_yaml(ablation_path).get("configs", {})
    configs = {}
    for name, overrides in experiment_settings.items():
        settings = _deep_merge(default_settings, overrides or {})
        configs[name] = _dict_to_config(settings)
    return configs
