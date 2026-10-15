# seekaudio: method-by-method reference

This atlas maps the production Python code to the [technical report](TECHNICAL_REPORT.md). Use the [learning guide](LEARNING_GUIDE.md) to study the prerequisites. Functions are listed by module, with their inputs, outputs, responsibilities, and important side effects.

The scope is the application, feature extractors, CLI, configuration classes, and deployment preparation script: **96 functions/classes**. Test functions are not production methods; the test suite is discussed in the technical report.

## 1. Start with the three call flows

### Offline catalog preparation

```text
__main__.main
  -> create_parser / load_config / execute_command
  -> cli_commands.run_all
  -> pipeline.run_pipeline
       1. ingest_dataset -> build_manifest -> create_manifest_record
       2. extract_dataset -> extract_feature_table
            -> extract_track_features -> load_audio
            -> extract_waveform_features -> individual extractors
       3. build_configured_space
            -> select_features -> build_feature_space
            -> standardize_features -> normalize_feature_groups
            -> apply_group_weights -> optional reduce_features_pca
            -> optional normalize_vectors
            -> attach_extraction_settings -> save_feature_space
       4. evaluate_precision_at_k -> find_similar_tracks
            -> calculate_ranking_metrics
            -> add_random_baseline -> save_evaluation
       5. run_ablations -> independent space fitting and evaluation
            -> plot_evaluation
```

### External audio analysis

```text
streamlit_app.main
  -> load_config / load_catalog
  -> select upload or sample / update_upload_state
  -> Analyze button -> analyze_upload
       -> analysis.analyze_audio
            -> read_audio_metadata
            -> load_audio
            -> extract_waveform_features
            -> search_feature_record
                 -> transform_features (NO FITTING)
                 -> find_similar_tracks
            -> add_track_details -> estimate_genre
            -> extract_librosa_characteristics
            -> render_mel_spectrogram
  -> show_analysis_result
       -> show_audio_characteristics / show_spectrogram
```

The bundled sample passes through the same analysis code using its existing local path. Uploads are written temporarily to supply a file path to the decoder and removed afterwards.

### CLI search

```text
search TRACK_ID
  -> run_search -> find_similar_tracks -> print_search_results

search-file FILE
  -> run_audio_search -> find_existing_track_id
  -> search_audio_file -> extract_track_features
  -> search_feature_record -> transform_features -> find_similar_tracks
  -> print_search_results
```

Genre labels are attached **after** acoustic ranking. They are not inputs to cosine similarity or PCA.

## 2. Entry point: [__main__.py](../src/seekaudio/__main__.py)

| Method | Input → output | Responsibility |
|---|---|---|
| `create_parser` | No input → `ArgumentParser` | Defines commands, option names, defaults, and help. Describing an argument does not execute a pipeline stage. |
| `execute_command` | Parsed arguments, configuration → `None` | Dispatches the selected command to its handler. Keeps command selection separate from implementation. |
| `main` | Optional list of command arguments → `None` | Parses arguments, loads configuration, dispatches, and handles expected command errors. The module entry guard calls it when run with `python -m`. |

**Learn here:** imports, functions, command-line parsing, `None`, exceptions, the entry-point guard.

## 3. Command handlers: [cli_commands.py](../src/seekaudio/cli_commands.py)

| Method | Input → output | Responsibility |
|---|---|---|
| `get_match_count` | Arguments, configuration → integer K | Uses the explicit command K when supplied, otherwise the configured K. Validation occurs downstream. |
| `print_search_results` | Ranked rows, manifest → `None` | Adds track details and prints a readable search table. This changes presentation, not rankings. |
| `run_ingest` | Arguments, configuration → `None` | Resolves audio/output paths, creates the manifest, and prints the saved track count. |
| `run_extract` | Arguments, configuration → `None` | Loads the manifest and extracts the configured raw feature table with its JSON recipe. |
| `run_build_space` | Arguments, configuration → `None` | Loads raw features, applies CLI representation overrides, builds the space, attaches extraction settings, and saves paired artifacts. A components value of zero selects no PCA. |
| `run_search` | Arguments, configuration → `None` | Loads saved catalog vectors and metadata, searches by an existing ID, and prints results. |
| `find_existing_track_id` | Audio path, manifest → ID or `None` | Compares resolved file paths to identify a catalog-file query. It does not compare audio contents or recognize copied recordings. |
| `run_audio_search` | Arguments, configuration → `None` | Loads vectors/models/manifest, identifies an exact existing path for optional self-exclusion, extracts and searches the file, and prints matches. |
| `run_evaluate` | Arguments, configuration → `None` | Loads saved artifacts, evaluates all catalog queries, adds random expectations, and writes reports. |
| `run_comparison` | Arguments, configuration → `None` | Routes comparison commands to the earlier group comparison or the YAML ablation workflow. |
| `run_all` | Arguments, configuration → `None` | Calls the offline pipeline with configuration and ablation paths. |

**Side effects:** printing, loading artifacts, and saving outputs for commands that build/evaluate. Search commands do not refit models.

## 4. Configuration: [config.py](../src/seekaudio/config.py)

### Configuration dataclasses

A dataclass is a convenient Python container with named fields. It is not an ML model.

| Class | What it contains | Why it exists |
|---|---|---|
| `AudioConfig` | Sample rate, excerpt duration, mono flag | Describes waveform preparation. |
| `FeatureGroupsConfig` | MFCC/spectral/rhythm/energy/tonal switches | Selects columns used by the search representation. |
| `FeatureWeightsConfig` | One numeric weight for each group | Controls post-group-normalization multipliers. |
| `FeaturesConfig` | FFT/hop sizes, MFCC/Mel/chroma counts, nested groups/weights | Keeps extraction and feature-selection settings together. Nested factories avoid shared mutable objects. |
| `PoolingConfig` | `use_std` | Controls whether standard-deviation columns are selected; extraction still computes them. |
| `SpaceConfig` | PCA component count, whitening, final L2 switch | Configures vector-space construction. |
| `RetrievalConfig` | Default K | Sets search/evaluation neighbor count. |
| `PathsConfig` | Audio, manifest, feature, space, evaluation paths and feature version | Defines artifact locations and versioned raw-feature filenames. |
| `SeekAudioConfig` | All nested configuration objects | Gives methods readable access such as `config.audio.sample_rate`. |

### Configuration methods

| Method | Input → output | Responsibility |
|---|---|---|
| `read_yaml` | Path → dictionary | Reads UTF-8 YAML with `safe_load`; an empty document becomes an empty dictionary. |
| `_deep_merge` | Base dictionary, override dictionary → merged dictionary | Recursively merges nested values without changing the original base. Scalars/lists replace earlier values. |
| `_dict_to_config` | Nested settings dictionary → `SeekAudioConfig` | Converts dictionaries to dataclasses, retaining defaults for unspecified fields. Copies feature settings before removing nested groups/weights. |
| `load_config` | YAML path, optional overrides → configuration | Checks file existence, reads settings, optionally merges overrides, and constructs named settings. |
| `load_ablation_configs` | Default and ablation YAML paths → name-to-configuration dictionary | Builds an independent complete configuration for each named experiment. |

An underscore prefix indicates an internal helper by convention; it is not a security boundary.

## 5. Ingestion: [ingest.py](../src/seekaudio/ingest.py)

| Method | Input → output | Responsibility |
|---|---|---|
| `find_audio_files` | Root directory → sorted list of paths | Recursively discovers supported extensions and checks the dataset directory. Sorting stabilizes track order. |
| `read_audio_metadata` | Audio path → metadata dictionary | Reads original sample rate, channel count, sample-frame count, and duration from SoundFile. Does not compute MFCCs. |
| `create_manifest_record` | Audio path → manifest record | Combines metadata with filename-stem ID, parent-directory genre, and path. |
| `build_manifest` | Dataset directory → DataFrame | Reads records, warns/skips unreadable files, and validates usable track identities. |
| `save_manifest` | Manifest, destination → destination Path | Creates the parent folder and saves Parquet without a pandas index column. |
| `ingest_dataset` | Dataset and output paths → DataFrame | Coordinates manifest construction and persistence. |

**Distinction:** original metadata describes the source recording; prepared-waveform measurements describe the normalized excerpt.

## 6. Offline orchestration: [pipeline.py](../src/seekaudio/pipeline.py)

| Method | Input → output | Responsibility |
|---|---|---|
| `write_json` | Path, dictionary → `None` | Writes indented UTF-8 JSON. The parent folder must already exist. |
| `feature_path` | Configuration → Path | Constructs `features_v2.parquet` or another configured feature-version filename. |
| `extraction_settings` | Configuration → extractor-argument dictionary | Validates mono and 12-bin chroma support. Always enables rhythm extraction for later ablations; tonal extraction follows configuration. |
| `extract_dataset` | Manifest, configuration, destination → feature DataFrame | Rejects an empty manifest, extracts each track, saves Parquet, and writes the adjacent extraction-recipe JSON. |
| `build_configured_space` | Raw table, configuration → (vectors, models) | Selects groups/pooling, calls the fitting pipeline with weights/PCA/L2 settings, and attaches the complete configuration to models. |
| `attach_extraction_settings` | Model dictionary, feature-file path → `None` | Mutates models by attaching recipe/version from adjacent JSON when it exists. Does not guess settings for legacy files. |
| `run_ablations` | Raw features, manifest, config paths, output, K → comparison DataFrame | Fits each experiment independently on the same existing feature table, scores retrieval, saves reports/configurations, and orders summaries by precision. No audio re-decoding. |
| `run_pipeline` | Configuration and config paths → per-track evaluation DataFrame | Executes ingestion, extraction, representation fitting, evaluation, and ablations/plots. Writes outputs but only reads raw recordings. |

**Important:** rerunning a build replaces artifacts at the configured output paths. Changing feature preparation requires rebuilding compatible downstream artifacts.

## 7. Waveform preparation: [features/audio.py](../src/seekaudio/features/audio.py)

| Method | Input → output | Responsibility |
|---|---|---|
| `load_audio` | Path, target rate/duration, mono flag → waveform array | Decodes, validates, converts channels when requested, rationally resamples, and center-crops or symmetrically pads. Normal pipeline output is a mono float32 vector of 661,500 samples. |
| `split_into_frames` | Waveform, frame length, hop → frame-first matrix | Creates complete overlapping frames. Short inputs are padded enough for one frame; the ordinary tail beyond the final complete frame is not another partial frame. Default excerpt yields (1288, 2048). |
| `calculate_power_spectrum` | Frame matrix → power matrix | Applies a symmetric Hann window, computes one-sided real FFT, and squares magnitudes. Default shape is (1288, 1025). |

See report Sections 6–7 for resampling, Nyquist, frame-count, Hann, FFT, and power equations.

## 8. Feature coordination: [features/extract.py](../src/seekaudio/features/extract.py)

| Method | Input → output | Responsibility |
|---|---|---|
| `extract_track_features` | Audio path and extraction settings → dictionary | Loads the prepared waveform and calls the waveform-level coordinator. ID initially comes from filename stem. |
| `extract_waveform_features` | Prepared mono waveform, ID, settings → dictionary | Combines MFCC, spectral, and energy dictionaries, plus optional rhythm/tonal. Supports waveform reuse in live analysis. |
| `extract_feature_table` | Manifest, settings, progress switch → DataFrame | Iterates paths, extracts records, restores the manifest's authoritative ID, and optionally prints progress every 100 tracks. Errors propagate rather than silently dropping extracted rows. |
| `save_feature_table` | Feature table, destination → Path | Creates directories and persists raw numeric measurements plus ID as Parquet. |

Default record: `track_id` plus **53** values. Default table: **999 × 54**. Extraction does not use genre labels to calculate the measurements.

## 9. Individual feature modules

### [features/energy.py](../src/seekaudio/features/energy.py)

| Method | Input → output | Responsibility |
|---|---|---|
| `extract_energy_features` | Waveform, frame/hop settings → two-value dictionary | Computes frame RMS from unwindowed samples and pools its mean/population standard deviation. |

### [features/spectral.py](../src/seekaudio/features/spectral.py)

| Method | Input → output | Responsibility |
|---|---|---|
| `extract_spectral_features` | Waveform, rate/frame/hop settings → eight-value dictionary | Pools mean/std of magnitude-weighted centroid, bandwidth, 85% magnitude rolloff, and frame zero-crossing rate. Handles silent spectral denominators. |

### [features/mfcc.py](../src/seekaudio/features/mfcc.py)

| Method | Input → output | Responsibility |
|---|---|---|
| `_create_mel_filter_bank` | Rate, FFT size, Mel count → filter matrix | Places triangular filters uniformly in HTK Mel coordinates from zero to Nyquist. Default matrix: (128, 1025). No Slaney area normalization. |
| `calculate_mel_power` | Waveform and spectrum/filter settings → Mel-power matrix | Frames, computes power, and multiplies by the Mel bank. Matrix is frame-first: (1288, 128) by default. |
| `extract_mfcc_features` | Waveform and MFCC settings → 40-value dictionary | Converts Mel power to logarithmic dB values with a floor/range clipping, takes orthonormal DCT-II, keeps 20 coefficients, and pools mean/std. |

**Naming trap:** `mfcc_01` corresponds to mathematical coefficient index zero, not index one.

### [features/rhythm.py](../src/seekaudio/features/rhythm.py)

| Method | Input → output | Responsibility |
|---|---|---|
| `extract_rhythm_features` | Waveform and framing/rate settings → three-value dictionary | Forms positive RMS changes, smooths them, pools onset mean/std, and estimates tempo from autocorrelation within 40–200 BPM. Weak/nonperiodic signals can return zero tempo. |

This is the stored retrieval tempo, not the UI's separate Librosa tempo.

### [features/tonal.py](../src/seekaudio/features/tonal.py)

| Method | Input → output | Responsibility |
|---|---|---|
| `extract_tonal_features` | Waveform and framing/rate settings → 36-value dictionary | Aggregates FFT power into 12 rounded pitch classes, normalizes chroma, projects to six tonal-centroid axes, and pools mean/std. |

Optional, disabled by default. Does not implement harmonic separation, instrument recognition, or a calibrated key detector.

### [features/librosa_analysis.py](../src/seekaudio/features/librosa_analysis.py)

| Method | Input → output | Responsibility |
|---|---|---|
| `extract_librosa_characteristics` | Prepared waveform and analysis settings → three-value dictionary | Transposes shared power into frequency-first form, computes spectral-flux tempo, mean flatness, and mean spectral contrast. Adapts contrast octave count to Nyquist. All-silent spectrum returns zeros. |

**Boundary:** display-only dictionary; these three values do not alter the existing retrieval feature schema, PCA, or genre vote.

### [features/spectrogram.py](../src/seekaudio/features/spectrogram.py)

| Method | Input → output | Responsibility |
|---|---|---|
| `compute_mel_spectrogram` | Prepared waveform and Mel settings → dB image matrix | Reuses custom Mel power, transposes to (bands, frames), and produces peak-relative display dB with an 80 dB range. |

Default image matrix: **128 × 1288**. Relative normalization means independently generated images do not preserve absolute recording-level differences.

## 10. Representation fitting/inference: [space.py](../src/seekaudio/space.py)

| Method | Input → output | Responsibility |
|---|---|---|
| `get_numeric_features` | Feature table → numeric-column DataFrame | Requires ID, removes it from calculations, and rejects nonfinite values. |
| `create_feature_table` | Values, output column names, source table → DataFrame | Wraps transformed values while preserving source index and IDs. |
| `normalize_rows` | Numeric matrix → numeric matrix | Divides each row by Euclidean norm. Zero norms are replaced by one, preserving zero rows. |
| `get_group_columns` | Table and group name → column-name list | Uses prefixes such as `mfcc_` to locate a group. |
| `standardize_features` | Catalog feature table → (standardized table, scaler) | Fits `StandardScaler` on catalog columns and returns the reusable scaler. Constant columns become zero. |
| `normalize_feature_groups` | Standardized table → table | Normalizes each nonzero group separately for each track and rejects unknown feature prefixes. Returns a copy. |
| `apply_group_weights` | Group-normalized table and weights → table | Checks recognized groups and finite nonnegative weights, then multiplies each group. Missing weights leave groups unchanged. |
| `reduce_features_pca` | Weighted table, components, whitening → (component table, PCA) | Fits full-SVD PCA. Limits dimensions to min(track count minus one, feature count) and rejects whitening negligible-variance components. |
| `normalize_vectors` | Numeric-feature/component table → table | Applies whole-vector L2 normalization, retaining IDs and zeros. Distinct from per-group normalization. |
| `build_feature_space` | Selected catalog features, representation options → (vectors, model dictionary) | Validates IDs and runs the complete fitting sequence. Saves scaler/PCA, weights, input/output column order, and L2 setting. |
| `save_feature_space` | Vectors, models, directory → Path | Saves `vectors.parquet` and `models.joblib`. Only load trusted serialized models. |
| `select_features` | Raw table, enabled groups, std switch → table | Selects group columns and optionally removes `_std` columns. Requires at least one available enabled group. |
| `transform_features` | Raw query rows, saved models → vector table | Reorders required columns and applies saved scaler, group processing, saved PCA, and saved final normalization. Extra columns are ignored; missing required columns fail. **Never fits.** |

The fitting/inference split is one of the project's most important architectural ideas. A query standardized against itself would not share the catalog coordinate system.

Default output: **999 × 41**, including ID and 40 PCA components. Do not interchange vectors and models from different fitting runs.

## 11. Retrieval: [retrieval.py](../src/seekaudio/retrieval.py)

| Method | Input → output | Responsibility |
|---|---|---|
| `find_similar_tracks` | Vector table, query ID, K → ranked ID/similarity table | Validates identities/numbers, rejects zero query direction, compares cosine against all rows, removes self and zero candidates, clips roundoff, and stably sorts descending. Returns up to K. |
| `add_track_details` | Ranked rows, manifest → enriched rows | Joins genre/path by ID after ranking, validating unique identities and complete matches. |
| `search_audio_file` | File, catalog vectors/models, K, optional exclusion ID → ranked rows | Requires saved extraction settings, extracts a record, and delegates to record search. |
| `search_feature_record` | Extracted dictionary, catalog artifacts, K, optional exclusion → ranked rows | Applies saved transforms, checks vector schema, creates an unused temporary query ID, appends a query to a copy, and reuses the common ranker. |

Self-ID exclusion does not detect other copies of the same recording. Similarity is a signed score, not a confidence percentage.

## 12. Live analysis: [analysis.py](../src/seekaudio/analysis.py)

| Method | Input → output | Responsibility |
|---|---|---|
| `estimate_genre` | Labeled nearest matches → genre string | Chooses most votes, then greatest summed similarity for vote ties. Exact remaining ties follow grouped genre order. Rejects empty/missing labels. |
| `analyze_audio` | File, vectors, models, manifest → analysis dictionary | Reads original metadata, loads one prepared waveform, extracts compatible retrieval features, searches ten neighbors, votes genre, computes sample peak/display-only Librosa descriptors, and renders a spectrogram PNG. |

Output contains original duration/rate/channels; excerpt-based technical measurements; analysis duration; genre; PNG bytes. A silent peak is `None` rather than a finite dBFS value. UI code handles formatting.

This method is not training a supervised classifier. The neighbor list is used internally even though it is hidden in the simplified interface.

## 13. Evaluation: [evaluate.py](../src/seekaudio/evaluate.py)

| Method | Input → output | Responsibility |
|---|---|---|
| `calculate_ranking_metrics` | Ordered binary relevance array, total relevant candidates → metric dictionary | Computes Precision@K, Recall@K, AP@K, and nDCG@K. AP denominator is min(returned count, total relevant). Requires a nonempty ranked array. |
| `evaluate_precision_at_k` | Vectors, labeled manifest, K → per-track DataFrame | Queries every catalog ID, excludes itself, compares result labels with query genre, and computes metrics. Validates K, labels, unique IDs, and nonzero vectors. |
| `add_random_baseline` | Fully evaluated per-track results → copied DataFrame | Adds exact expected random precision: (genre count minus one)/(catalog count minus one). No Monte Carlo sampling. |
| `save_evaluation` | Results, K, output directory → Path | Writes per-track CSV, per-genre CSV, and summary JSON. Saves optional metrics/baselines when supplied and computes lift when random expectation is positive. |

The method `evaluate_precision_at_k` retains its original name but now calculates four metrics. Mean AP over all query rows is reported as MAP@K.

## 14. Earlier comparisons: [experiments.py](../src/seekaudio/experiments.py)

| Method | Input → output | Responsibility |
|---|---|---|
| `compare_feature_groups` | Raw features, manifest, K, output → comparison DataFrame | Runs the earlier MFCC/spectral/energy group study, fits each representation, evaluates, and saves comparison outputs. |

The main `run` workflow now uses the eight YAML ablations through `pipeline.run_ablations`. Do not confuse the earlier comparison command with that configured experiment set.

## 15. Visualization: [viz.py](../src/seekaudio/viz.py)

| Method | Input → output | Responsibility |
|---|---|---|
| `plot_evaluation` | Per-track scores, ablation summary, directory, K → `None` | Saves horizontal genre/ablation precision bar charts. Genre chart includes random reference; axes span zero to one. Closes pyplot figures. |
| `render_mel_spectrogram` | Prepared waveform and settings → PNG bytes | Builds custom log-Mel data, positions frame/filter pixel centers, labels time and Mel-spaced Hz ticks, and renders a dark-theme image into memory using headless Matplotlib. |

The image time axis refers to the excerpt, not its offset in the full file. Color is peak-relative dB, not dBFS or calibrated sound-pressure level.

## 16. Web interface: [streamlit_app.py](../streamlit_app.py)

| Method | Input → output | Responsibility |
|---|---|---|
| `read_table` | Path string, modification timestamp → DataFrame | Cached Parquet loading via `cache_data`. Timestamp is part of the cache key even though the function body only uses the path. |
| `read_models` | Path string, modification timestamp → model dictionary | Cached trusted Joblib loading via `cache_resource`. Timestamp invalidates changed artifacts. |
| `load_catalog` | Configuration → (manifest, vectors, models) | Resolves paths relative to the app project root and loads the prebuilt catalog with modification-keyed caching. |
| `configure_page` | No input → `None` | Sets page configuration and CSS, including hiding the default top toolbar. Does not configure feature extraction. |
| `update_upload_state` | Audio bytes → `None` | Hashes selected content with an analysis-version prefix and resets stale session results when the selected audio changes. |
| `analyze_upload` | Upload, manifest, vectors, models → dictionary | Writes a decoder-readable temporary file, calls analysis, and guarantees cleanup. |
| `show_spectrogram` | Analysis dictionary → `None` | Displays PNG, caption, and download control. |
| `show_metric_row` | Measurement specifications → `None` | Creates columns and shows one row of formatted metrics/help text. |
| `show_audio_characteristics` | Analysis dictionary → `None` | Formats fourteen measurements and displays them in rows, explaining excerpt-based analysis. |
| `show_analysis_result` | Analysis dictionary → `None` | Shows genre estimate with its limitation, characteristics, and spectrogram. |
| `main` | No input → `None` | Loads catalog, offers upload/sample, previews audio, handles Analyze-button spinner/errors, stores results in session state, and displays existing results on reruns. |

**Two different forms of reuse:**

- Shared file/model caches avoid unnecessary artifact reloads.
- Session state keeps a particular visitor's selected audio and analysis result across Streamlit reruns.

Neither means new uploads are added to the catalog or used for model training.

## 17. Deployment preparation: [scripts/prepare_streamlit.py](../scripts/prepare_streamlit.py)

| Method | Input → output | Responsibility |
|---|---|---|
| `prepare_deployment` | No arguments → deployment-directory Path | Copies required source/config/dependencies and saved manifest/vectors/models to a small bundle, including the optional sample when present. Checks required inputs before copying. |

This prepares local files, not a hosted deployment. It does not push Git, choose a cloud branch, or remove the original repository's history. It leaves unrelated existing bundle files alone; rerunning can overwrite matching destination files.

## 18. Reading a function independently

For any method, answer these six questions in your notebook:

1. **Input:** Is this a file path, waveform, frame matrix, dictionary, or DataFrame?
2. **Shape/units:** What does each axis or column represent?
3. **Computation:** Is it reading, calculating, fitting, transforming, ranking, or displaying?
4. **Output:** What shape/type and units leave the method?
5. **Side effects:** Does it write disk, print, mutate a dictionary, or update UI/session state?
6. **Failure cases:** What happens for silence, missing columns, duplicate IDs, invalid settings, or unreadable audio?

Do not treat all `dict` values as equivalent: an extraction record, a model dictionary, and a display-analysis dictionary have different contracts.

## 19. Practical navigation exercises

### Exercise A: follow a catalog recording

Choose one manifest row. Locate its path and genre. Follow `extract_track_features`, count the 53 numeric fields, then follow `build_feature_space` to the 40 final components. Explain why genre is absent from the transformation.

### Exercise B: follow an upload

Set breakpoints at `analyze_audio`, `extract_waveform_features`, `transform_features`, and `estimate_genre`. Inspect shape, dtype, field names, and output type. Verify there is no `fit` call on this path.

### Exercise C: explain one wrong-looking result

If tempo appears half/double what you expect, trace the separate stored and displayed rhythm methods. Compare periodicity and listen. Avoid changing PCA models merely to adjust a display value.

### Exercise D: reproduce one evaluation row

Inspect ten ranked neighbors for a query. Create the binary genre-relevance sequence and manually calculate four ranking metrics. Compare with `calculate_ranking_metrics` and the saved CSV.

### Exercise E: explain an artifact mismatch

Describe why a vector file from one fitting run and a scaler/PCA dictionary from another may have identical column names yet incompatible coordinates. Explain why artifacts should be generated, stored, and deployed as a matched set.

Return to the [technical report](TECHNICAL_REPORT.md) for formulas and the [learning guide](LEARNING_GUIDE.md) for the reconstruction sequence.

