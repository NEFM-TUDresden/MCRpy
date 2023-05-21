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

from mcrpy.optimizers.SimulatedAnnealing import SimulatedAnnealing
from mcrpy.src import optimizer_factory

class YTPost(SimulatedAnnealing):

    def __init__(self, **kwargs):
        kwargs['mutation_rule'] = 'neighbor'
        kwargs['acceptance_distribution'] = 'zero_tolerance'
        super().__init__(**kwargs)

def register() -> None:
    optimizer_factory.register("YTPost", YTPost)
