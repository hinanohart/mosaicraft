"""Download demo assets for mosaicraft README figures.

Fetches a public-domain painting from Wikimedia Commons and a
pool of ~1024 CC0 photographs from picsum.photos (Unsplash-powered), then
writes ``docs/assets/MANIFEST.json`` with per-file SHA256 and license
metadata so the download is bit-exact reproducible.

Zundamon (the second demo target) ships committed in the repository under
``docs/assets/paintings/zundamon.jpg`` and is not downloaded — it is
included in the manifest for integrity verification only.

The downloaded image files are *not* committed (they live under
``docs/assets/`` which is gitignored); only ``zundamon.jpg`` and
``MANIFEST.json`` are versioned. A fresh clone can run::

    python scripts/download_demo_assets.py            # download + verify
    python scripts/download_demo_assets.py --verify-only  # integrity check
    python scripts/download_demo_assets.py --force    # force re-download
    python scripts/download_demo_assets.py --offline  # fail if any asset missing

Wikimedia paintings are public domain (pre-1929 or author died >70 years
ago). Zundamon is used under the Tohoku Zunko Guidelines. All Picsum tiles
are CC0 via the Unsplash license (free for any use, attribution appreciated
but not required).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "docs" / "assets"
PAINTINGS_DIR = ASSETS_DIR / "paintings"
TILES_DIR = ASSETS_DIR / "tiles"
MANIFEST_PATH = ASSETS_DIR / "MANIFEST.json"

USER_AGENT = "mosaicraft-demo-builder/1.0 (https://github.com/hinanohart/mosaicraft)"
WIKIMEDIA_DELAY_SEC = 1.0  # polite pause between Wikimedia hits
PICSUM_DELAY_SEC = 0.05  # picsum tolerates burst, but we rate-limit anyway
TILE_COUNT = 1024
TILE_PIXELS = 128
MAX_TILE_WORKERS = 8

# --- public-domain paintings from Wikimedia Commons ------------------------
#
# Pre-1929 original whose copyright has long expired.
# Wikimedia URL resolved via commons.wikimedia.org/w/api.php on 2026-04-09;
# SHA256 is recomputed at download time and pinned in MANIFEST.json.
# We fetch 2048px thumbnails instead of originals to keep the payload small.

PAINTINGS: list[dict[str, Any]] = [
    {
        "name": "pearl_earring.jpg",
        "title": "Girl with a Pearl Earring",
        "artist": "Johannes Vermeer",
        "year": "c. 1665",
        "license": "Public Domain (author died 1675, pre-1929 work)",
        "source_page": "https://commons.wikimedia.org/wiki/File:Meisje_met_de_parel.jpg",
        "url": (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/"
            "d/d7/Meisje_met_de_parel.jpg/"
            "2048px-Meisje_met_de_parel.jpg"
        ),
    },
]

# --- committed targets (not downloaded, already in git) ---------------------
#
# Zundamon is a character by SSS LLC / Tohoku Zunko Project.  The image is
# committed under docs/assets/paintings/ and covered by the Tohoku Zunko
# Guidelines (https://zunko.jp/guideline.html).  It is NOT auto-downloaded;
# this list exists so the manifest includes its SHA256 for verify-only runs.

COMMITTED_TARGETS: list[dict[str, Any]] = [
    {
        "name": "zundamon.jpg",
        "title": "Zundamon (illustration)",
        "artist": "SSS LLC / Tohoku Zunko Project",
        "year": "2024",
        "license": "Tohoku Zunko Guidelines (https://zunko.jp/guideline.html)",
        "source_page": "https://zunko.jp/guideline.html",
    },
]

TILE_SOURCE = {
    "provider": "picsum.photos",
    "upstream": "Unsplash Lite",
    "license": "Unsplash License (CC0-equivalent: free for any use, no attribution required)",
    "license_url": "https://unsplash.com/license",
    "url_template": f"https://picsum.photos/seed/mosaicraft{{seed}}/{TILE_PIXELS}",
    "count": TILE_COUNT,
    "size_px": TILE_PIXELS,
}


def sha256_of(path: Path) -> str:
    """Return hex SHA256 of the file at *path* (streaming, 1 MB chunks)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def http_get(url: str, dest: Path, timeout: float = 60.0) -> int:
    """Fetch *url* into *dest* with a custom User-Agent. Returns bytes written."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status} for {url}")
            data = resp.read()
        tmp.write_bytes(data)
        tmp.replace(dest)
        return len(data)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def load_existing_manifest() -> dict[str, Any]:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def download_paintings(*, force: bool, offline: bool) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for i, p in enumerate(PAINTINGS):
        dest = PAINTINGS_DIR / p["name"]
        if dest.exists() and not force:
            pass  # reuse
        else:
            if offline:
                raise SystemExit(f"--offline set but {dest} is missing")
            if i > 0:
                time.sleep(WIKIMEDIA_DELAY_SEC)
            print(f"  painting [{i + 1}/{len(PAINTINGS)}] {p['name']} ... ", end="", flush=True)
            try:
                size = http_get(p["url"], dest)
                print(f"{size / 1024:.1f} KB")
            except (urllib.error.URLError, RuntimeError) as e:
                print(f"FAIL ({e})")
                raise
        entries.append(
            {
                "path": f"paintings/{p['name']}",
                "sha256": sha256_of(dest),
                "size_bytes": dest.stat().st_size,
                "title": p["title"],
                "artist": p["artist"],
                "year": p["year"],
                "license": p["license"],
                "source_url": p["url"],
                "source_page": p["source_page"],
            }
        )

    # Committed targets (already in git, not downloaded).
    for ct in COMMITTED_TARGETS:
        dest = PAINTINGS_DIR / ct["name"]
        if not dest.exists():
            if offline:
                raise SystemExit(f"--offline set but committed target {dest} is missing")
            raise SystemExit(
                f"committed target {dest} not found — it should ship with the repository. "
                "Re-clone or run `git checkout -- docs/assets/paintings/zundamon.jpg`."
            )
        print(f"  committed target: {ct['name']} ({dest.stat().st_size / 1024:.1f} KB)")
        entries.append(
            {
                "path": f"paintings/{ct['name']}",
                "sha256": sha256_of(dest),
                "size_bytes": dest.stat().st_size,
                "title": ct["title"],
                "artist": ct["artist"],
                "year": ct["year"],
                "license": ct["license"],
                "source_page": ct["source_page"],
                "committed": True,
            }
        )
    return entries


def _fetch_tile(seed: int, force: bool) -> tuple[int, Path, int]:
    dest = TILES_DIR / f"tile_{seed:04d}.jpg"
    if dest.exists() and not force:
        return seed, dest, dest.stat().st_size
    url = TILE_SOURCE["url_template"].format(seed=seed)
    time.sleep(PICSUM_DELAY_SEC)
    size = http_get(url, dest, timeout=30.0)
    return seed, dest, size


def download_tiles(*, force: bool, offline: bool) -> list[dict[str, Any]]:
    TILES_DIR.mkdir(parents=True, exist_ok=True)
    # Fast path: all tiles present already and not forcing → just hash.
    expected = [TILES_DIR / f"tile_{i:04d}.jpg" for i in range(TILE_COUNT)]
    missing = [i for i, p in enumerate(expected) if not p.exists()]
    if missing and offline:
        raise SystemExit(f"--offline set but {len(missing)} tiles missing")
    if missing and not force:
        print(f"  tiles: {len(missing)} missing, downloading ...")
    elif force:
        print(f"  tiles: forcing re-download of {TILE_COUNT}")
        missing = list(range(TILE_COUNT))
    else:
        print(f"  tiles: all {TILE_COUNT} cached")

    if missing:
        done = 0
        with cf.ThreadPoolExecutor(max_workers=MAX_TILE_WORKERS) as pool:
            futures = {pool.submit(_fetch_tile, seed, force): seed for seed in missing}
            for fut in cf.as_completed(futures):
                seed = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    raise SystemExit(f"tile seed={seed} failed: {e}") from e
                done += 1
                if done % 64 == 0 or done == len(missing):
                    print(f"    {done}/{len(missing)} tiles", flush=True)

    entries: list[dict[str, Any]] = []
    for i, dest in enumerate(expected):
        entries.append(
            {
                "path": f"tiles/tile_{i:04d}.jpg",
                "sha256": sha256_of(dest),
                "size_bytes": dest.stat().st_size,
                "seed": i,
            }
        )
    return entries


def verify_against(manifest: dict[str, Any]) -> int:
    """Re-hash every file listed in *manifest* and report mismatches. Returns count."""
    mismatches = 0
    files = manifest.get("paintings", []) + manifest.get("tiles", [])
    for entry in files:
        dest = ASSETS_DIR / entry["path"]
        if not dest.exists():
            print(f"  MISSING: {entry['path']}")
            mismatches += 1
            continue
        actual = sha256_of(dest)
        if actual != entry["sha256"]:
            print(f"  MISMATCH: {entry['path']}  expected={entry['sha256'][:16]} got={actual[:16]}")
            mismatches += 1
    total = len(files)
    ok = total - mismatches
    print(f"  verify: {ok}/{total} OK, {mismatches} mismatched/missing")
    return mismatches


def write_manifest(paintings: list[dict[str, Any]], tiles: list[dict[str, Any]]) -> None:
    manifest = {
        "schema_version": 1,
        "generated_by": "scripts/download_demo_assets.py",
        "generated_on": time.strftime("%Y-%m-%d"),
        "tile_source": TILE_SOURCE,
        "paintings": paintings,
        "tiles": tiles,
        "totals": {
            "paintings": len(paintings),
            "tiles": len(tiles),
            "bytes": sum(e["size_bytes"] for e in paintings + tiles),
        },
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"  wrote {MANIFEST_PATH.relative_to(REPO_ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="re-hash files against MANIFEST.json and exit",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="fail instead of downloading missing files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.verify_only:
        existing = load_existing_manifest()
        if not existing:
            print("no MANIFEST.json found — run without --verify-only first")
            return 1
        print("verifying demo assets against MANIFEST.json ...")
        return 1 if verify_against(existing) else 0

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    print("processing target paintings ...")
    paintings = download_paintings(force=args.force, offline=args.offline)
    print("downloading CC0 tile pool from picsum.photos ...")
    tiles = download_tiles(force=args.force, offline=args.offline)
    print("writing manifest ...")
    write_manifest(paintings, tiles)

    total_mb = sum(e["size_bytes"] for e in paintings + tiles) / (1024 * 1024)
    print(f"\ndone. {len(paintings)} paintings + {len(tiles)} tiles, total {total_mb:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
