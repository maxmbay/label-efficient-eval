"""Load Criteo Uplift v2.1 (download script; raw CSV never committed).

Source: http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz
Criteo card: 25,309,483 rows. TFDS ``criteo`` split lists 13,979,592; verify
on load and record both in the load report.
"""

from __future__ import annotations

import csv
import gzip
import json
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Prefer the Hugging Face mirror; fall back to the historical Criteo redirect.
CRITEO_URLS = (
    "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz",
    "http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz",
)
CRITEO_URL = CRITEO_URLS[0]
CRITEO_CARD_N_ROWS = 25_309_483
TFDS_TRAIN_N_ROWS = 13_979_592
FEATURE_COLS = tuple(f"f{i}" for i in range(12))
REQUIRED_COLS = FEATURE_COLS + (
    "treatment",
    "conversion",
    "visit",
    "exposure",
)
DEFAULT_SUBSAMPLE_N = 500_000
SEED_ROOT = 20261026


@dataclass(frozen=True)
class CriteoLoadReport:
    n_rows_file: int
    n_rows_used: int
    subsampled: bool
    subsample_n: int | None
    seed: int
    treatment_rate: float
    visit_rate: float
    conversion_rate: float
    criteo_card_n_rows: int
    tfds_train_n_rows: int
    row_count_matches_criteo_card: bool
    notes: tuple[str, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class CriteoLoadError(RuntimeError):
    """Raised when columns or download fail."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_cache_dir() -> Path:
    return _repo_root() / "data" / "raw" / "criteo"


def download_criteo(
    *,
    cache_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """Download the gzipped CSV to cache; return the local path."""
    cache = cache_dir or default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / "criteo-research-uplift-v2.1.csv.gz"
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None
    for url in CRITEO_URLS:
        try:
            print(f"Downloading Criteo uplift from {url} ...")
            urllib.request.urlretrieve(url, tmp)  # noqa: S310 (fixed URLs)
            if tmp.stat().st_size < 1_000_000:
                raise CriteoLoadError(
                    f"Download from {url} looks too small "
                    f"({tmp.stat().st_size} bytes)."
                )
            tmp.replace(dest)
            return dest
        except Exception as exc:  # noqa: BLE001 (try next mirror)
            last_err = exc
            print(f"Failed: {exc}")
            if tmp.exists():
                tmp.unlink()
    raise CriteoLoadError(f"All Criteo download mirrors failed: {last_err}")


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise CriteoLoadError(f"Missing columns: {missing}")
    out = df.copy()
    for c in FEATURE_COLS:
        out[c] = out[c].astype(float)
    out["treatment"] = out["treatment"].astype(int)
    for c in ("conversion", "visit", "exposure"):
        out[c] = out[c].astype(int)
    return out


def _reservoir_sample(
    path: Path,
    *,
    k: int,
    seed: int,
) -> tuple[pd.DataFrame, int]:
    """One-pass Algorithm R over the gzip CSV; returns (frame, n_file)."""
    rng = np.random.default_rng(seed)
    reservoir: list[list[str]] = []
    n_seen = 0
    with gzip.open(path, "rt", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            n_seen += 1
            if len(reservoir) < k:
                reservoir.append(row)
            else:
                j = int(rng.integers(0, n_seen))
                if j < k:
                    reservoir[j] = row
    if n_seen < k:
        raise CriteoLoadError(
            f"File has only {n_seen} rows; cannot sample {k}."
        )
    df = pd.DataFrame(reservoir, columns=header)
    return df, n_seen


def load_criteo(
    *,
    cache_dir: Path | None = None,
    force_download: bool = False,
    subsample_n: int | None = DEFAULT_SUBSAMPLE_N,
    seed: int = SEED_ROOT,
    full: bool = False,
) -> tuple[pd.DataFrame, CriteoLoadReport]:
    """Load Criteo uplift; by default a deterministic subsample of 500k rows.

    Set ``full=True`` to read every row (needs several GB RAM). Prefer that on
    Databricks; document any subsample in run notes.
    """
    path = download_criteo(cache_dir=cache_dir, force=force_download)
    notes: list[str] = [
        f"Downloaded/cached at {path}.",
        f"Criteo AI Lab card claims {CRITEO_CARD_N_ROWS:,} rows; "
        f"TFDS criteo train split lists {TFDS_TRAIN_N_ROWS:,}.",
    ]

    use_full = full or subsample_n is None
    cache = cache_dir or default_cache_dir()

    if use_full:
        df = _normalize(pd.read_csv(path))
        n_file = len(df)
        used = n_file
        subsampled = False
        sub_n: int | None = None
        notes.append(f"Observed file data rows: {n_file:,}.")
        notes.append("Loaded full file into pandas.")
    else:
        k = int(subsample_n)
        sub_path = cache / f"subsample_{k}_seed{seed}.parquet"
        meta_path = cache / f"subsample_{k}_seed{seed}_meta.json"
        if sub_path.exists() and meta_path.exists() and not force_download:
            df = _normalize(pd.read_parquet(sub_path))
            meta = json.loads(meta_path.read_text())
            n_file = int(meta["n_rows_file"])
            used = len(df)
            subsampled = True
            sub_n = k
            notes.append(f"Loaded cached subsample from {sub_path}.")
            notes.append(f"Observed file data rows (from meta): {n_file:,}.")
        else:
            print(
                f"Reservoir-sampling {k:,} rows from Criteo "
                f"(seed={seed}); one pass over the gzip..."
            )
            df_raw, n_file = _reservoir_sample(path, k=k, seed=seed)
            df = _normalize(df_raw)
            used = len(df)
            subsampled = True
            sub_n = k
            df.to_parquet(sub_path, index=False)
            meta_path.write_text(
                json.dumps({"n_rows_file": n_file, "k": k, "seed": seed}) + "\n"
            )
            notes.append(f"Observed file data rows: {n_file:,}.")
            notes.append(
                f"Deterministic reservoir subsample of {used:,} rows with "
                f"seed={seed} (local / Databricks-limits path)."
            )
            notes.append(f"Cached subsample at {sub_path}.")

    matches_card = n_file == CRITEO_CARD_N_ROWS
    if not matches_card:
        notes.append(
            "File row count does not match the Criteo card; proceeding with "
            "the observed count and recording the discrepancy."
        )
    if n_file != TFDS_TRAIN_N_ROWS:
        notes.append(
            "File row count differs from the TFDS train split size "
            f"({TFDS_TRAIN_N_ROWS:,}); TFDS may subsample or shard."
        )

    report = CriteoLoadReport(
        n_rows_file=n_file,
        n_rows_used=used,
        subsampled=subsampled,
        subsample_n=sub_n,
        seed=seed,
        treatment_rate=float(df["treatment"].mean()),
        visit_rate=float(df["visit"].mean()),
        conversion_rate=float(df["conversion"].mean()),
        criteo_card_n_rows=CRITEO_CARD_N_ROWS,
        tfds_train_n_rows=TFDS_TRAIN_N_ROWS,
        row_count_matches_criteo_card=matches_card,
        notes=tuple(notes),
    )
    return df, report


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return df.loc[:, list(FEATURE_COLS)].to_numpy(dtype=float)
