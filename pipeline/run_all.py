"""Run the full pipeline end-to-end: parse → features → graph."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    ("parse PPTX", ROOT / "parse_pptx.py"),
    ("compute features", ROOT / "compute_features.py"),
    ("build graph", ROOT / "build_graph.py"),
]


def main() -> int:
    for label, script in STEPS:
        print(f"\n=== {label} ({script.name}) ===")
        result = subprocess.run([sys.executable, str(script)])
        if result.returncode != 0:
            print(f"Step '{label}' failed.", file=sys.stderr)
            return result.returncode
    print("\nPipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
