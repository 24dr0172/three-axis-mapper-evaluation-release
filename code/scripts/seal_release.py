#!/usr/bin/env python3
"""Regenerate the top-level release SHA-256 manifest deterministically."""

from __future__ import annotations

import hashlib
import pathlib


RELEASE = pathlib.Path(__file__).resolve().parents[2]
MANIFEST = RELEASE / "SHA256SUMS.txt"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    residue = [
        path for path in RELEASE.rglob("*")
        if path.is_file()
        and (path.suffix in {".pyc", ".orig", ".bak"} or "__pycache__" in path.parts)
    ]
    if residue:
        names = "\n".join(path.relative_to(RELEASE).as_posix() for path in residue[:20])
        raise SystemExit(f"refusing to seal with build/backup residue:\n{names}")

    files = sorted(
        (path for path in RELEASE.rglob("*") if path.is_file() and path != MANIFEST),
        key=lambda path: path.relative_to(RELEASE).as_posix(),
    )
    lines = [
        f"{sha256(path)}  {path.relative_to(RELEASE).as_posix()}"
        for path in files
    ]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"sealed {len(files)} release files in {MANIFEST.relative_to(RELEASE)}")


if __name__ == "__main__":
    main()
