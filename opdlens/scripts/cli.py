"""The ``opdlens`` CLI: launch / eval / schema.

(Offline artifact fitting is run directly: ``python -m opdlens.fit.fit_jlens`` /
``fit_bridge``.)
"""

from __future__ import annotations

import argparse

from ..utils.config import add_config_patch_arguments


def main() -> None:
    parser = argparse.ArgumentParser(prog="opdlens")
    sub = parser.add_subparsers(dest="command", required=True)

    p_launch = sub.add_parser("launch", help="Launch training via torchrun.")
    p_launch.add_argument("config")
    add_config_patch_arguments(p_launch)

    p_eval = sub.add_parser(
        "eval", help="Evaluate a checkpoint or a bare model (no training)."
    )
    p_eval.add_argument("config")
    p_eval.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint dir / state.pt whose student weights to load; omit to eval "
        "the config's model as-is (e.g. point `student` at the teacher).",
    )
    add_config_patch_arguments(p_eval)

    p_schema = sub.add_parser("schema", help="Generate the config JSON schema.")
    p_schema.add_argument("--output-dir", default="schema")

    args = parser.parse_args()
    if args.command == "launch":
        from . import launch

        launch.run(args.config, args.config_updates, args.config_removes)
    elif args.command == "eval":
        from . import eval as eval_cmd

        eval_cmd.run(
            args.config, args.checkpoint, args.config_updates, args.config_removes
        )
    elif args.command == "schema":
        from . import schema

        schema.run(args.output_dir)


if __name__ == "__main__":
    main()
