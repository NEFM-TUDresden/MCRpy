"""
Function for checking the connectivity/percolation of, e.g.
(a) pores with gas channel
(b) material with electrolyte

Three different labels [1]:
    (i)   ISOLATED  - clusters that lack contact to the connected phase of the reconstruction
    (ii)  UNKNOWN   - clusters appear isolated, but intersect one of the ofter faces of the volume (or they are connected only by an edge/vertex)
    (iii) CONNECTED - connected to reconstruction's electrolyte's face or current collector/gas channel face or to both faces
                      (depending whether cluster is ionic, electronic or mixed conducting or porosity)

in other words:
- LSCF connected when they percolate with electrolyte and current collector
- for electron conducting materials: need to be percolated to current collector
- for ion conducting materials: need to be percolated to electrolyte
- pores: need to be connected to gas channel

[1] Joos 2016: Dissertation
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label  # , generate_binary_structure
from typing import Union


import tensorflow as tf

from mcrpy.src import descriptor_factory
from mcrpy.descriptors.PhaseDescriptor3D import PhaseDescriptor3D


class Percolation3D(PhaseDescriptor3D):
    """3D Percolation, see module docstring"""

    is_differentiable = False

    @staticmethod
    def make_singlephase_descriptor(**kwargs) -> callable:
        sides_to_check = (("left", "right"), ("top", "bottom"), ("back", "front"))

        # @tf.function
        def compute_descriptor(microstructure) -> np.ndarray:
            if microstructure.shape[0] == 1:
                microstructure = microstructure[0]
            if microstructure.shape[-1] == 1:
                microstructure = microstructure[..., 0]
            return np.stack(
                [
                    np.stack(percolation_3D(microstructure, side_to_check=side_to_check))
                    for side_to_check in sides_to_check
                ]
            )

        return compute_descriptor


def register() -> None:
    descriptor_factory.register("Percolation3D", Percolation3D)


# does not work with numba.njit -> "TypingError: Cannot determine Numba type of <class 'type'>" -> error araises at valid_keywords
def percolation_3D(array: np.ndarray, side_to_check: Union[str, tuple[str]] = None, structure_label: np.ndarray = None):
    """
    Function to check the connectivity of a phase (e.g. pores) to a fuel cell element (e.g. gas channel)

    Parameters
    - array : Numpy array of the detailed microstructure
    - side_to_check : Either string or tuple with sides (left, right, top, bottom)
                      Default: None; but this is actually NOT useful
                      Only combinations with left/right, top/bottom, back/front can be considered
    structure_label : input of the scipy.ndimage.label-function
                      Determines which connectivity counts.
                      Default: None -> it only considers features to be connected if the voxels touch each other through faces

    Returns
    f_connected : fraction of connected pixels
    f_isolated : fraction of isolated pixels
    f_unknown : fraction of unknown pixels
    """

    # check whether input is right
    valid_keywords = {"left", "right", "top", "bottom", "front", "back"}
    valid_combinations = {frozenset(("left", "right")), frozenset(("top", "bottom")), frozenset(("back", "front"))}

    if side_to_check is not None:
        if isinstance(side_to_check, str):
            if side_to_check not in valid_keywords:
                raise ValueError(
                    f"Invalid value for 'side_to_check': {side_to_check}. "
                    f"Valid values are: {', '.join(valid_keywords)}."
                )
        elif isinstance(side_to_check, tuple):
            sides_set = frozenset(side_to_check)
            if sides_set not in valid_combinations:
                raise ValueError(
                    f"Invalid combination in 'side_to_check': {side_to_check}. "
                    f"Valid combinations are: {', '.join(map(str, valid_combinations))}."
                )
        else:
            raise ValueError("Invalid value for 'side_to_check' parameter.")

    # label array
    array_labeled, n_labels = label(array, structure=structure_label)

    # initialize empty object for classification of all clusters
    classif_percol = np.zeros(n_labels, dtype=object)
    # total number of pixels/voxels of phase i
    n_pixels = np.count_nonzero(array == 1)
    # initialize number of connected and isolated pixels
    n_connected = np.array(0)
    n_isolated = np.array(0)

    # check whether cluster is connected to the respective side
    for i in range(n_labels):
        # Extract the binary mask for the current labeled region
        cluster_mask = array_labeled == i + 1

        # Check cluster connections based on the specified side(s)
        if side_to_check is None:
            # No specific side specified, consider all sides
            side_connected = (
                np.any(cluster_mask[:, :, 0])
                and np.any(cluster_mask[:, :, -1])
                and np.any(cluster_mask[:, 0, :])
                and np.any(cluster_mask[:, -1, :])
                and np.any(cluster_mask[0, :, :])
                and np.any(cluster_mask[-1, :, :])
            )
        elif isinstance(side_to_check, str):
            # Single side specified
            side_connected = (
                (side_to_check == "left" and np.any(cluster_mask[:, :, 0]))
                or (side_to_check == "right" and np.any(cluster_mask[:, :, -1]))
                or (side_to_check == "top" and np.any(cluster_mask[:, 0, :]))
                or (side_to_check == "bottom" and np.any(cluster_mask[:, -1, :]))
                or (side_to_check == "front" and np.any(cluster_mask[0, :, :]))
                or (side_to_check == "back" and np.any(cluster_mask[-1, :, :]))
            )
        elif isinstance(side_to_check, tuple) and len(side_to_check) == 2:
            # Two sides specified in a tuple
            side_connected = (
                ("left" in side_to_check and np.any(cluster_mask[:, :, 0]))
                and ("right" in side_to_check and np.any(cluster_mask[:, :, -1]))
                or ("top" in side_to_check and np.any(cluster_mask[:, 0, :]))
                and ("bottom" in side_to_check and np.any(cluster_mask[:, -1, :]))
                or ("front" in side_to_check and np.any(cluster_mask[0, :, :]))
                and ("back" in side_to_check and np.any(cluster_mask[-1, :, :]))
            )
        else:
            raise ValueError("Invalid value for 'side_to_check' parameter.")

        # update the percolated array and count pixel classifications
        if side_connected:
            classif_percol[i] = "Connected"
            n_connected += np.sum(cluster_mask)
        elif (
            np.any(cluster_mask[0, :, :])
            or np.any(cluster_mask[-1, :, :])
            or np.any(cluster_mask[:, 0, :])
            or np.any(cluster_mask[:, -1, :])
            or np.any(cluster_mask[:, :, 0])
            or np.any(cluster_mask[:, :, -1])
        ):
            classif_percol[i] = "Unknown"
        else:
            classif_percol[i] = "Isolated"
            n_isolated += np.sum(cluster_mask)

    f_connected = n_connected / n_pixels
    f_isolated = n_isolated / n_pixels
    f_unknown = (n_pixels - n_connected - n_isolated) / n_pixels

    return f_connected, f_isolated, f_unknown
