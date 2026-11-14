"""Export everything the website needs as static files, so it can be hosted without a server.

Writes to ``frontend/public/data/``:

    datasets.json                         list of datasets (only the exported subject)
    datasets/<dataset>/<subject>.json     dataset metadata (same schema as GET /api/datasets/{id})
    filters/<dataset>_<subject>_<fs>.ssfp filter packs at 44.1 kHz (native) and 48 kHz (fallback)
    sounds.json, sounds/*.wav             built-in sounds marked publishable in the manifest
    NOTICE.txt                            copyright notice required by the CIPIC licence

Run (from backend/):  python -m sonicsphere.export_static [--subject 165] [--include-unpublishable]
Then build the site:  cd ../frontend && npm run build
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .config import get_settings
from .filter_pack import get_pack

OUT = Path(__file__).resolve().parents[2] / "frontend" / "public" / "data"


def main(argv=None) -> int:
    settings = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=settings.default_dataset)
    ap.add_argument("--subject", default=None)
    ap.add_argument("--rates", nargs="*", type=int, default=[44100, 48000],
                    help="sampling rates to export (the first is used when the browser accepts it)")
    ap.add_argument("--include-unpublishable", action="store_true",
                    help="also export sounds whose licence does not allow public hosting (e.g. the TTS voice)")
    args = ap.parse_args(argv)

    cfg = settings.dataset(args.dataset)
    subject = args.subject or cfg.default_subject
    from .main import dataset_info  # same metadata as the API (validates the dataset too)

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "datasets" / cfg.id).mkdir(parents=True)
    (OUT / "filters").mkdir()
    (OUT / "sounds").mkdir()

    info = dataset_info(cfg.id, subject).model_dump()
    info["subjects"] = [s for s in info["subjects"] if s["id"] == subject]
    (OUT / "datasets" / cfg.id / f"{subject}.json").write_text(json.dumps(info), encoding="utf-8")
    (OUT / "datasets.json").write_text(json.dumps([{
        "id": cfg.id, "name": cfg.name, "default_subject": subject, "downloaded_subjects": [subject],
        "listed_subjects": 1}]), encoding="utf-8")

    total = 0
    for fs in args.rates:
        data, _ = get_pack(cfg.id, subject, float(fs), settings)
        name = f"{cfg.id}_{subject}_{fs}.ssfp"
        (OUT / "filters" / name).write_bytes(data)
        total += len(data)
        print(f"  filters/{name}  {len(data) / 1e6:.2f} MB")

    manifest = json.loads((settings.sounds_dir / "manifest.json").read_text(encoding="utf-8"))
    sounds = []
    for s in manifest:
        if not s.get("publishable", True) and not args.include_unpublishable:
            print(f"  skipped sound '{s['id']}' (not cleared for public hosting: {s['license']})")
            continue
        src = settings.sounds_dir / s["file"]
        shutil.copyfile(src, OUT / "sounds" / s["file"])
        total += src.stat().st_size
        sounds.append(s | {"url": f"sounds/{s['file']}"})
    (OUT / "sounds.json").write_text(json.dumps(sounds, ensure_ascii=False, indent=1), encoding="utf-8")

    (OUT / "NOTICE.txt").write_text(
        f"{cfg.name} - subject {subject}\n{cfg.reference}\n\n"
        "The filter files in filters/ are derived from the CIPIC HRTF Database.\n"
        f"Copyright notice and licence (from the original files):\n{info['license_text']}\n\n"
        "Built-in sounds: see sounds.json for the source and licence of each file.\n", encoding="utf-8")
    print(f"Wrote {OUT}  ({total / 1e6:.1f} MB, {len(sounds)} sounds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
