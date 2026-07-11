"""Merge Jacobian-lens fits over disjoint calibration shards."""

from __future__ import annotations

import argparse

from ..jlens import JacobianLens
from ..utils.logging import console


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    merged = JacobianLens.merge([JacobianLens.load(path) for path in args.inputs])
    merged.save(args.out)
    console.print(f"[green]Saved merged Jacobian lens to {args.out}[/green] {merged!r}")


if __name__ == "__main__":
    main()
