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
from dataclasses import dataclass, field
import logging
from typing import List

@dataclass
class CommonSettings:
    data_folder: str = None
    target_folder: str = None
    precision: int = 64
    information: str = None
    logfile_name: str = 'logfile'
    logging_level: int = logging.INFO
    logfile_date: bool = False
    func: callable = None

@dataclass
class DescriptorSettings:
    descriptor_types: List[str] = field(default_factory=lambda: ['Correlations'])
    slice_mode: str = 'average'
    nl_method: str = 'relu'
    limit_to: int = 16
    threshold_steepness: float = 10.0
    use_multiphase: bool = True
    grey_values: bool = False
    use_multigrid_descriptor: bool = True
    isotropic: bool = False
    gram_weights_filename: str = 'vgg19_normalized.pkl'

@dataclass
class OptimizerSettings:
    optimizer_type: str = 'LBFGSB'
    learning_rate: float = 0.01
    beta_1: float = 0.9
    beta_2: float = 0.999
    rho: float = 0.95
    momentum: float = 0.0
    initial_temperature: float = 0.004
    final_temperature: float = None
    cooldown_factor: float = 0.9

@dataclass
class LossSettings:
    loss_type: str = 'MSE'
    descriptor_weights: List[float] = None
    oor_multiplier: float = 1000.0
    phase_sum_multiplier: float = 1000.0

@dataclass
class CharacterizationSettings(CommonSettings, DescriptorSettings):
    microstructure_filename: str = None

@dataclass
class ReconstructionSettings(CommonSettings, OptimizerSettings, LossSettings, DescriptorSettings):
    descriptor_filename: str = None
    max_iter: int = 500
    stride: int = 4
    convergence_data_steps: int = 50
    outfile_data_steps: int = None
    tolerance: float = 1e-12
    use_multigrid_reconstruction: bool = False

@dataclass
class MatchingSettings(CharacterizationSettings, ReconstructionSettings):
    add_dimension: int = None

def select_subsettings(from_settings, to_settings_class):
    from_settings_dict = vars(from_settings)
    dummy_settings_dict = vars(to_settings_class())
    for k, v in dummy_settings_dict.items():
        if k in from_settings_dict:
            dummy_settings_dict[k] = from_settings_dict[k]
    return to_settings_class(**dummy_settings_dict)
