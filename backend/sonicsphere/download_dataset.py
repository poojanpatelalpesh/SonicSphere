"""Download HRIR datasets from the SOFA conventions repository.

Examples (run from ``backend/``)::

    python -m sonicsphere.download_dataset                       # CIPIC default subjects
    python -m sonicsphere.download_dataset --subjects 003 165
    python -m sonicsphere.download_dataset --all                 # all 45 CIPIC subjects (~80 MB)
    python -m sonicsphere.download_dataset --dataset mit_kemar --subjects normal_pinna

Every file is checked (HDF5 signature, SOFA conventions, loadable by the adapter) and its
SHA-256 is written to ``<dataset dir>/MANIFEST.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

from .config import DatasetConfig, get_settings
from .dataset_loader import DatasetError, file_sha256, load_sofa

USER_AGENT = "SonicSphere-dataset-fetcher/1.0"


def _download(url: str, dest: Path, retries: int = 3) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as out:
                while chunk := resp.read(1 << 16):
                    out.write(chunk)
            tmp.replace(dest)
            return
        except Exception as exc:  # network errors are retried
            last = exc
            time.sleep(1.5 * attempt)
    if tmp.exists():
        tmp.unlink()
    raise RuntimeError(f"Failed to download {url}: {last}")


def fetch_subject(cfg: DatasetConfig, subject: str, force: bool = False) -> dict:
    if not cfg.download_base_url:
        raise RuntimeError(f"Dataset '{cfg.id}' has no download_base_url configured")
    cfg.directory.mkdir(parents=True, exist_ok=True)
    dest = cfg.subject_path(subject)
    remote_name = dest.name
    if cfg.id == "mit_kemar":  # files are published as mit_kemar_<pinna>.sofa
        remote_name = f"mit_kemar_{subject}.sofa"
    url = cfg.download_base_url.rstrip("/") + "/" + remote_name
    if dest.exists() and not force:
        status = "present"
    else:
        print(f"  downloading {url}")
        _download(url, dest)
        status = "downloaded"
    try:
        ds = load_sofa(dest, dataset_id=cfg.id, subject=subject)
    except DatasetError as exc:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"{dest.name} failed validation and was removed: {exc}") from exc
    return {
        "subject": subject,
        "file": dest.name,
        "url": url,
        "sha256": file_sha256(dest),
        "bytes": dest.stat().st_size,
        "fs": ds.fs,
        "directions": ds.num_directions,
        "hrir_length": ds.hrir_length,
        "status": status,
    }


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=settings.default_dataset, choices=sorted(settings.datasets))
    ap.add_argument("--subjects", nargs="*", help="subject ids (default: the dataset's default_download list)")
    ap.add_argument("--all", action="store_true", help="download every listed subject")
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args(argv)

    cfg = settings.dataset(args.dataset)
    subjects = list(cfg.subjects) if args.all else (args.subjects or list(cfg.default_download)
                                                     or [cfg.default_subject])
    unknown = [s for s in subjects if cfg.subjects and s not in cfg.subjects]
    if unknown:
        print(f"Unknown subject ids for {cfg.id}: {unknown}", file=sys.stderr)
        return 2

    manifest_path = cfg.directory / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": {}}
    print(f"Dataset {cfg.name} -> {cfg.directory}")
    failures = 0
    for sid in subjects:
        try:
            info = fetch_subject(cfg, sid, force=args.force)
        except Exception as exc:
            failures += 1
            print(f"  [error] {sid}: {exc}", file=sys.stderr)
            continue
        manifest["files"][sid] = info
        print(f"  [ok] {sid}: {info['directions']} directions, {info['hrir_length']} taps @ "
              f"{info['fs']:.0f} Hz, sha256 {info['sha256'][:16]}... ({info['status']})")
    manifest.update({"dataset": cfg.id, "name": cfg.name, "source": cfg.download_base_url,
                     "updated": time.strftime("%Y-%m-%dT%H:%M:%S")})
    cfg.directory.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
