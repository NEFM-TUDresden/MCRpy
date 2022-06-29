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
from typing import Any, Callable, Dict, Tuple, Union

from mcrpy.losses.Loss import Loss

loss_creation_functions: Dict[str, Callable[..., Loss]] = {}
loss_choices = [d[:-3] for d in os.listdir(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'losses')) 
    if d.endswith('.py') and d not in {'Loss.py', '__init__.py'}]


def register(loss_type: str, creator_fn: Callable[..., Loss]) -> None:
    loss_creation_functions[loss_type] = creator_fn


def unregister(loss_type: str) -> None:
    loss_creation_functions.pop(loss_type, None)


def create(loss_type: str, arguments: Dict[str, Any]) -> Union[Loss, Tuple[Loss]]:
    assert 'anisotropic' in arguments.keys()
    if arguments['anisotropic']:
        losses = []
        for dim in range(3):
            args_copy = arguments.copy()
            d_des_lst = [d_des[dim] for d_des in arguments['desired_descriptor_list']]
            args_copy['desired_descriptor_list'] = d_des_lst
            losses.append(try_create(loss_type, args_copy))
        return tuple(losses)
    else:
        return try_create(loss_type, arguments)

def try_create(loss_type: str, arguments: Dict[str, Any]) -> Loss:
    try:
        creator_func = loss_creation_functions[loss_type]
    except KeyError:
        raise ValueError(f"Unknown loss type {loss_type!r}") from None
    args_copy = arguments.copy()
    return creator_func(**args_copy)

