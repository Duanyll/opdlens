"""Config-file loading + JSONPath overrides (adapted from flow_control).

``load_config_file`` reads json/json5/jsonc (also yaml/toml), strips a top-level
``$schema`` key, and applies ``--update JSONPATH=VALUE`` / ``--remove JSONPATH``
patches so overrides can be forwarded across the process boundary (launch parent
→ torchrun child, which re-loads the same file).
"""

import argparse
import contextlib
import tomllib
from collections.abc import Sequence
from typing import Any

import json5
import yaml
from jsonpath_ng.ext import parse


def add_config_patch_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--update",
        dest="config_updates",
        action="append",
        default=[],
        metavar="JSONPATH=VALUE",
        help=(
            "Update all config nodes matched by JSONPath. VALUE is parsed as a "
            "JSON5 literal when possible, otherwise as a string. Can be repeated."
        ),
    )
    parser.add_argument(
        "--remove",
        dest="config_removes",
        action="append",
        default=[],
        metavar="JSONPATH",
        help="Remove all config nodes matched by JSONPath. Can be repeated.",
    )


def format_config_patch_args(
    updates: Sequence[str] = (), removes: Sequence[str] = ()
) -> list[str]:
    """Render config patches back into ``--update``/``--remove`` CLI args.

    Used to forward overrides to subprocesses that re-load the same config file
    (e.g. ``launch`` spawning the torchrun child).
    """
    args: list[str] = []
    for update in updates:
        args.extend(["--update", update])
    for remove in removes:
        args.extend(["--remove", remove])
    return args


def _find_update_separator(update: str) -> int | None:
    quote: str | None = None
    escaped = False
    bracket_depth = 0
    for i, char in enumerate(update):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
            continue
        if char == "[":
            bracket_depth += 1
            continue
        if char == "]" and bracket_depth:
            bracket_depth -= 1
            continue
        if char == "=" and bracket_depth == 0:
            return i
    return None


def _split_update(update: str) -> tuple[str, str]:
    separator = _find_update_separator(update)
    if separator is None:
        raise ValueError(
            f"Config update must be formatted as JSONPATH=VALUE: {update!r}"
        )
    path = update[:separator].strip()
    value = update[separator + 1 :].strip()
    if not path:
        raise ValueError(f"Empty JSONPath in config update: {update!r}")
    return path, value


def _parse_update_value(value: str) -> Any:
    try:
        return json5.loads(value)
    except ValueError:
        return value


def _require_matches(config: Any, expression: str) -> Any:
    path = parse(expression)
    if not path.find(config):
        raise ValueError(f"JSONPath matched nothing: {expression}")
    return path


def apply_config_patches(
    config: dict[str, Any], updates: Sequence[str] = (), removes: Sequence[str] = ()
) -> dict[str, Any]:
    for update in updates:
        expression, value_text = _split_update(update)
        # update_or_create (not strict .update): lets --update SET a key the config
        # file omits but the model defines (e.g. run_id). Genuinely unknown keys are
        # still caught by the models' extra="forbid" validation downstream.
        parse(expression).update_or_create(config, _parse_update_value(value_text))
    for expression in removes:
        _require_matches(config, expression).filter(lambda _: True, config)
    return config


def load_config_file(
    path: str, updates: Sequence[str] = (), removes: Sequence[str] = ()
) -> dict:
    """Load a JSON/JSON5/JSONC (or YAML/TOML) config file into a plain dict."""
    if path.endswith((".json", ".json5", ".jsonc")):
        with open(path, encoding="utf-8") as f:
            res = json5.load(f)
    elif path.endswith((".yaml", ".yml")):
        with open(path, encoding="utf-8") as f:
            res = yaml.safe_load(f)
    elif path.endswith(".toml"):
        with open(path, "rb") as f:
            res = tomllib.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {path}")

    if not isinstance(res, dict):
        raise ValueError(f"Config file must contain an object: {path}")
    with contextlib.suppress(KeyError):
        del res["$schema"]

    apply_config_patches(res, updates, removes)
    return res
