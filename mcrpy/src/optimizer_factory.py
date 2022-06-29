"""
   Copyright 10/2020 - 04/2021 Paul Seibert for Diploma Thesis at TU Dresden
   Copyright 05/2021 - 12/2021 TU Dresden (Paul Seibert as Scientific Assistant)
   Copyright 2022 TU Dresden (Paul Seibert as Scientific Employee)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, Set

from mcrpy.optimizers.Optimizer import Optimizer

optimizer_classes: Dict[str, Callable[..., Optimizer]] = {}
optimizer_choices = [d[:-3] for d in os.listdir(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'optimizers')) 
    if d.endswith('.py') and d not in {'Optimizer.py', 'TFOptimizer.py', 
        'SPOptimizer.py', '__init__.py'}]


def register(optimizer_type: str,
        creator_fn: Callable[..., Optimizer]) -> None:
    optimizer_classes[optimizer_type] = creator_fn


def unregister(optimizer_type: str) -> None:
    optimizer_classes.pop(optimizer_type, None)


def create(optimizer_type: str,
        arguments: Dict[str, Any]) -> Optimizer:
    try:
        creator_func = optimizer_classes[optimizer_type]
    except KeyError:
        raise ValueError(f"Unknown optimizer type {optimizer_type!r}") from None
    args_copy = arguments.copy()
    return creator_func(**args_copy)

