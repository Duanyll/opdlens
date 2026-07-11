"""Generate the OpdTrainer config JSON schema (jsonc-friendly)."""

from __future__ import annotations

import json
from pathlib import Path

from ..utils.logging import console


def run(output_dir: str = "schema") -> None:
    from pydantic import TypeAdapter

    from ..training import OpdTrainer

    schema = TypeAdapter(OpdTrainer).json_schema()
    # ``$schema`` is stripped before validation (load_config_file), but declaring
    # it keeps editors happy; allowTrailingCommas enables jsonc.
    schema.setdefault("properties", {}).setdefault("$schema", {"type": "string"})
    schema["allowTrailingCommas"] = True

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "opd.schema.json"
    path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    console.print(f"[green]Wrote schema to {path}[/green]")


if __name__ == "__main__":
    run()
