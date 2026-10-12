# seekaudio

seekaudio is a Python audio-analysis and music-retrieval project built to explore
the intersection of AI, analytics, and audio signal processing. It compares songs
using measurable sound characteristics rather than artist names or listening history.

## Project overview

We built a complete workflow that prepares an audio dataset, extracts musical
features, creates a searchable feature space, and evaluates the retrieved matches.
The local Streamlit demo lets visitors upload audio or try the supplied
"Sweet Child O' Mine" sample. It displays an estimated genre, 14 technical audio
measurements, and a downloadable Mel spectrogram of the analyzed excerpt.

What has been implemented:

- Dataset ingestion and validation: a saved manifest of 999 usable GTZAN tracks,
  with one unreadable recording skipped.
- Audio preparation: mono conversion, resampling to 22,050 Hz, and a centered
  30-second excerpt with silence padding for shorter files.
- Feature extraction: 53 numeric values covering MFCC timbre, spectral properties,
  RMS energy, and rhythm, with optional tonal features.
- Similarity search: standardization, group normalization and weighting, optional
  PCA and whitening, and cosine-similarity ranking. Uploaded queries reuse the
  saved transforms.
- Evaluation: Precision@K, Recall@K, MAP@K, nDCG@K, an expected random baseline,
  eight feature ablations, saved reports, and comparison plots.
- A simple web demo: estimated genre from nearest-match voting, audio preview,
  technical measurements, and a Mel spectrogram using the same extraction settings.
  Librosa supplies three display analyses: spectral-flux tempo estimation,
  spectral flatness, and spectral contrast.
- Reproducible command-line stages, saved models, and automated tests.

The current combined-feature PCA pipeline achieves **Precision@10 of 0.4862**,
compared with **0.0991** for expected random retrieval. MFCC-only retrieval scores
**0.5027**, showing that adding more features does not automatically improve results.
These measure same-genre retrieval within the catalog, not held-out classification
accuracy. The demo's genre estimate is based on neighbor votes, not a trained,
calibrated genre classifier.

## Setup

Use Python 3.12 or newer. Run commands from the project directory.

```powershell
python -m pip install -e ".[dev]"
```

Place audio in `data/raw`, arranged as `genre/track.wav`. The manifest obtains
genre from the parent folder and track ID from the filename without its extension.
Track IDs must be unique. WAV, FLAC, and MP3 support depends on the local decoder.
Audio data is not downloaded by the application.

## One command

```powershell
python -m src.seekaudio run
```

This scans audio, saves the manifest, extracts version 2 features, builds the
search space, evaluates Precision@10, runs eight ablations, and creates PNG plots.
It prints extraction progress every 100 tracks. Each run rebuilds its output
files. Existing version 1 feature and baseline reports remain available.

```powershell
python -m src.seekaudio search blues.00000 --k 5
python -m src.seekaudio search-file "C:\music\query.wav" --k 5
python -m streamlit run streamlit_app.py
python -m pytest tests -q
```

After installation, `python -m seekaudio` or `seekaudio` also works.

The Streamlit demo accepts uploads or a bundled sample and displays an estimated genre and technical
audio measurements. Genre is predicted by majority vote among up to ten nearest
catalog vectors, with summed similarity breaking vote ties. This is not a trained
or calibrated genre classifier. Retrieval results remain available through the CLI.

To enable the sample song option, place your audio file at
`data/demo/sweet_child_o_mine.mp3`. The sample is supplied locally; the app does
not download recordings. Only bundle recordings you have permission to redistribute
when publishing the demo. Without that file, the upload option still works.

## Deploy to Streamlit Community Cloud

Create a small deployment folder with the saved catalog and optional sample:

```powershell
python scripts/prepare_streamlit.py
```

The output is `deployment/streamlit/`. Upload its contents to a separate GitHub
repository, keeping `streamlit_app.py` at the repository root. The bundle includes
the Python source, pinned dependencies, Streamlit settings, manifest, vectors,
models, and locally supplied sample. It excludes the raw dataset, feature tables,
tests, caches, and experiment reports. No dataset rebuilding is needed in the cloud.

The original repository already tracks `data/raw/`; `.gitignore` does not remove
previously tracked files or Git history. The separate deployment repository avoids
copying that large dataset and does not alter your existing Git index or files.

On [Streamlit Community Cloud](https://share.streamlit.io), click **Create app**,
select the deployment repository and branch, set the main file to
`streamlit_app.py`, choose **Python 3.12** under advanced settings, and deploy.
The app uses `requirements.txt`; its pinned scikit-learn version matches the saved
model. Uploads are limited to 25 MB. Secrets are not required for this app.

Only include the sample recording in a public repository if you have permission
to redistribute it. The upload workflow works without a bundled sample.

Official guides: [deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
and [dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

## Reading and running individual stages

For a first read of the Python code, follow this order:

1. `src/seekaudio/ingest.py`: find songs and create their metadata table.
2. `src/seekaudio/features/audio.py`: load audio and split it into short windows.
3. `src/seekaudio/features/extract.py`: combine the feature extractors.
4. `src/seekaudio/space.py`: scale and transform the feature table.
5. `src/seekaudio/retrieval.py`: compare songs and rank matches.
6. `src/seekaudio/analysis.py`: estimate genre and prepare display measurements.
7. `streamlit_app.py`: display uploads and analysis results.

`__main__.py` defines command arguments and chooses a command handler.
`cli_commands.py` runs the individual commands. `pipeline.py` connects the stages.

```powershell
python -m src.seekaudio ingest
python -m src.seekaudio extract
python -m src.seekaudio build-space
python -m src.seekaudio evaluate
python -m src.seekaudio ablate
```

Every stage accepts `--config config/default.yaml`. Use `--help` for path and K
overrides. To skip PCA: `build-space --components 0`. The `run` command reads all
settings from YAML, including feature groups, weights, pooling, PCA, and paths.
Standalone `compare-features` retains the earlier MFCC/spectral/energy comparison;
`ablate` includes rhythm and the configured eight experiments.

## Outputs

| File | Contents |
|---|---|
| `data/manifest.parquet` | Track IDs, genre, paths, and technical metadata |
| `data/features/features_v2.parquet` | 53 numeric features plus track ID |
| `data/features/features_v2.json` | Extraction recipe and version |
| `data/space/vectors.parquet` | Dataset vectors |
| `data/space/models.joblib` | Fitted transforms and query extraction recipe |
| `data/evaluation/current/` | Per-track, per-genre, and overall scores |
| `data/evaluation/ablations/` | Experiment scores and settings |
| `data/evaluation/plots/` | Genre and ablation charts |

The default features contain 40 MFCC, 8 spectral, 2 RMS energy, and 3 rhythm
values. Optional tonal extraction adds 12 chroma and 6 tonal-centroid dimensions,
each summarized by mean and standard deviation.

## Method and limitations

Audio is converted to mono at 22,050 Hz and center-cropped or symmetrically padded
to 30 seconds. MFCCs use a NumPy power spectrum, triangular HTK Mel filters,
decibel conversion with an 80 dB range, and an orthonormal type-II DCT. These are
not numerically interchangeable with Librosa's default Slaney MFCC settings.
Spectral features use magnitude weighting. Rhythm uses positive RMS changes and
autocorrelation within 40–200 BPM; half-time and double-time ambiguity remains.
Tonal features use FFT pitch-class aggregation, not harmonic source separation.

The UI tempo is estimated separately with Librosa from a log-power spectral-flux
onset envelope. It also displays Librosa spectral flatness and spectral contrast.
These display measurements do not replace the stored version 2 retrieval features,
so the existing scaler, PCA model, evaluation scores, and dataset vectors remain
compatible. MFCC extraction and the displayed Mel spectrogram retain their custom
NumPy implementation.

Numeric columns are standardized, groups normalized separately, weighted,
optionally reduced and whitened with PCA, then optionally L2 normalized. Cosine
similarity ranks candidates. External queries use the saved dataset transforms;
they never refit the scaler or PCA. Only load model files produced by this project
or another trusted source, since Joblib uses pickle serialization.

Evaluation excludes the query and measures same-genre Precision@K. The exact
uniform-random baseline accounts for genre counts. Transforms are fitted on the
same catalog being evaluated; these are descriptive catalog results, not held-out
generalization scores. GTZAN genre labels are imperfect similarity judgments,
and duplicates can inflate scores. One local track, `jazz.00054.wav`, is unreadable
and skipped, leaving 999 usable tracks. Results need listening checks and a
separate evaluation dataset before broader claims about musical similarity.

Reports also include Recall@K, mean average precision at K, and nDCG@K. AP@K
divides by the smaller of K and the number of relevant catalog tracks; nDCG uses
binary same-genre relevance. Tracks without any relevant candidates receive zero.
