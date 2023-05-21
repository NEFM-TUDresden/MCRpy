import numpy as np

from pymks import TwoPointCorrelation
import mcrpy

cutoff = 25
limit_to = cutoff + 1
ms = np.load('/home/paul/Dokumente/Code/pyMCR/microstructures/pymks_ms_64x64.npy')

pymks_correlations = TwoPointCorrelation(correlations=[(1, 1)], cutoff=cutoff, periodic_boundary=True).transform(
        np.expand_dims(np.stack([1-ms, ms], axis=-1), axis=0))

settings = mcrpy.CharacterizationSettings(
    descriptor_types=['TwoPointCorrelations'], limit_to=limit_to, use_multigrid_descriptor=False, use_multiphase=False)
mcrpy_correlations = mcrpy.characterize(ms, settings)['TwoPointCorrelations']

def compute_factor(settings: mcrpy.CharacterizationSettings) -> float:
    zl = 1 / (1 + np.exp(0.75 * settings.threshold_steepness))
    zu = 1 / (1 + np.exp(-(0.25 * settings.threshold_steepness) ))
    a = 1 / (zu - zl)
    b = -a * zl
    sigmoid = lambda z: 1/(1 + np.exp(-z))
    return sigmoid(-0.25 * settings.threshold_steepness) * a + b

def correct_thresholding(pcs: np.array, settings: mcrpy.CharacterizationSettings) -> np.array:
    ft05 = compute_factor(settings)
    factor = 2 * ft05 ** 2
    pcs[1:] = (1 - factor) * pcs[1:] + factor * pcs[0]
    return pcs

def sort_pymks_to_mcrpy(pymks_s2: np.ndarray, limit_to: int) -> np.ndarray:
    assert pymks_s2.shape[0] == 1
    assert pymks_s2.shape[-1] == 1
    assert len(pymks_s2.shape) == 4
    pymks_s2 = pymks_s2[0, ..., 0]

    out = []
    for i in range(limit_to):
        for j in range(limit_to):
            out.append(pymks_s2[cutoff + i, cutoff + j])
    for i in range(1, limit_to):
        for j in range(1, limit_to):
            out.append(pymks_s2[cutoff + i, cutoff - j])
    return np.array(out)

def convert_pymks_to_mcrpy(pymks_correlations: np.ndarray, settings: mcrpy.CharacterizationSettings) -> np.ndarray:
    return correct_thresholding(sort_pymks_to_mcrpy(pymks_correlations, settings.limit_to), settings)

def error(a, b):
    return np.max(np.abs(a - b))

print(error(convert_pymks_to_mcrpy(pymks_correlations, settings), mcrpy_correlations))

