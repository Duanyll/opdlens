"""Optimizer / scheduler dict-config (adapted from flow_control.utils.types).

A bare ``{"class_name": "AdamW", "lr": 2e-6}`` dict resolves to the torch class by
name; every other key is forwarded as a constructor kwarg. Not a discriminated
union — just a schema-annotated dict + a ``parse_*`` resolver.
"""

from __future__ import annotations

from typing import Annotated, Any

import torch
from pydantic import WithJsonSchema

OptimizerConfig = Annotated[
    dict[str, Any],
    WithJsonSchema(
        {
            "type": "object",
            "properties": {
                "class_name": {
                    "type": "string",
                    "description": "torch.optim class name (e.g. AdamW).",
                },
                "lr": {"type": "number", "description": "Learning rate."},
            },
            "required": ["class_name"],
            "additionalProperties": True,
        }
    ),
]


def parse_optimizer(conf: OptimizerConfig, parameters: Any) -> torch.optim.Optimizer:
    conf = dict(conf)
    class_name = conf.pop("class_name")
    ctor = getattr(torch.optim, class_name)
    return ctor(parameters, **conf)


SchedulerConfig = Annotated[
    dict[str, Any],
    WithJsonSchema(
        {
            "type": "object",
            "properties": {
                "class_name": {
                    "type": "string",
                    "description": "torch.optim.lr_scheduler class name (e.g. ConstantLR).",
                }
            },
            "required": ["class_name"],
            "additionalProperties": True,
        }
    ),
]


def parse_scheduler(conf: SchedulerConfig, optimizer: torch.optim.Optimizer) -> Any:
    conf = dict(conf)
    class_name = conf.pop("class_name")
    ctor = getattr(torch.optim.lr_scheduler, class_name)
    return ctor(optimizer, **conf)
