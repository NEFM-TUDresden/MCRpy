"""
Written by Eric Langner at TU Dresden
"""

import numpy as np
from scipy.ndimage import label, generate_binary_structure

# possible inputs for structuring elements
##########################################

# structuring element that will consider features connected even if they only touch diagonally
struc_1_2D = generate_binary_structure(2, 2)
struc_1_3D = generate_binary_structure(3, 3)
# structuring element that will consider features connected only if they touch by faces
struc_2_2D = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])  # also default for label function
struc_2_3D = np.array(
    [[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [1, 1, 1], [0, 1, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]
)  # also default for label function


def detect_islands(array):
    """
    Function to detect islands as material inclusions in pore are not possible

    Parameters
    ----------
    array : voxel array that is filled with 1 (material) or 0 (pore)

    Returns
    -------
    array : boolean array with True values for island voxels

    """

    # get labeled array (all islands have different numbers except pores?)
    labeled_array, num_features = label(array)
    # delete zeros from labeled array (do not need to be changed  in postprocessing)
    labeled_array_without_0 = np.delete(labeled_array.ravel(), np.where(labeled_array.ravel() == 0))
    # get unique labels and count them
    unique, counts = np.unique(labeled_array_without_0, return_counts=True)
    # get largest connected area (do not need to be changed in postprocessing)
    connected_area = int(np.where(counts == max(counts))[0])

    # get island labels without label for largest connected area
    island_labels_mask = np.in1d(unique, unique[connected_area], invert=True)
    island_labels = unique[island_labels_mask]

    # set boolean array entries to True for islands
    array_boolean = np.isin(labeled_array, island_labels)

    return array_boolean


def voxel_periodicity(array):
    """
    Function to determine the periodicity of the array
    -> it marks every voxel with "TRUE" that is NOT periodic
    Parameters
    ----------
    array : numpy array

    Returns
    -------
    is_periodic : boolean array
        gives True values for the voxels of the boundary that are not periodic.

    """

    num_dims = array.ndim
    shape = array.shape

    # for 2 dimensions
    if num_dims == 2:
        is_periodic = np.zeros(shape, dtype=bool)
        is_periodic[0, :] = array[0, :] != array[-1, :]  # Top boundary
        is_periodic[-1, :] = array[-1, :] != array[0, :]  # Bottom boundary
        is_periodic[:, 0] = array[:, 0] != array[:, -1]  # Left boundary
        is_periodic[:, -1] = array[:, -1] != array[:, 0]  # Right boundary

        # corners
        is_periodic[0, 0] = np.logical_or.reduce(
            [(array[0, 0] != array[0, -1]), (array[0, 0] != array[-1, 0]), (array[0, 0] != array[-1, -1])]
        )  # corners
        is_periodic[-1, 0] = is_periodic[0, 0]
        is_periodic[0, -1] = is_periodic[0, 0]
        is_periodic[-1, -1] = is_periodic[0, 0]

        if is_periodic[0, 0] == True:
            print("Corners are not periodic")
        else:
            pass

    # for three dimensions
    elif num_dims == 3:
        is_periodic = np.zeros(shape, dtype=bool)
        is_periodic[0, :, :] = array[0, :, :] != array[-1, :, :]  # Front boundary
        is_periodic[-1, :, :] = array[-1, :, :] != array[0, :, :]  # Back boundary
        is_periodic[:, 0, :] = array[:, 0, :] != array[:, -1, :]  # Left boundary
        is_periodic[:, -1, :] = array[:, -1, :] != array[:, 0, :]  # Right boundary
        is_periodic[:, :, 0] = array[:, :, 0] != array[:, :, -1]  # Top boundary
        is_periodic[:, :, -1] = array[:, :, -1] != array[:, :, 0]  # Bottom boundary

        # corners
        is_periodic[0, 0, 0] = np.logical_or.reduce(
            [
                (array[0, 0, 0] != array[0, -1, 0]),
                (array[0, 0, 0] != array[-1, 0, 0]),
                (array[0, 0, 0] != array[0, 0, -1]),
                (array[0, 0, 0] != array[-1, 0, -1]),
                (array[0, 0, 0] != array[0, -1, -1]),
                (array[0, 0, 0] != array[-1, -1, 0]),
                (array[0, 0, 0] != array[-1, -1, -1]),
            ]
        )
        for j in [0, -1]:
            for k in [0, -1]:
                for l in [0, -1]:
                    is_periodic[j, k, l] = is_periodic[0, 0, 0]

        if is_periodic[0, 0, 0] == True:
            print("Corners are not periodic")
        else:
            pass

    else:
        raise ValueError("Array dimension not supported. Only 2D and 3D arrays are supported.")

    if np.any(is_periodic):
        print("Array is not periodic")
    else:
        print("Array is periodic")

    return is_periodic


def mark_boundary_voxels(array):
    shape = array.shape
    num_dims = array.ndim
    boundary_boolarray = np.zeros(shape, dtype=bool)

    # for 2 dimensions
    if num_dims == 2:
        boundary_boolarray[0, :] = True  # Front boundary
        boundary_boolarray[-1, :] = True  # Back boundary
        boundary_boolarray[:, 0] = True  # Left boundary
        boundary_boolarray[:, -1] = True  # Right boundary

    # for three dimensions
    elif num_dims == 3:
        boundary_boolarray[0, :, :] = True  # Front boundary
        boundary_boolarray[-1, :, :] = True  # Back boundary
        boundary_boolarray[:, 0, :] = True  # Left boundary
        boundary_boolarray[:, -1, :] = True  # Right boundary
        boundary_boolarray[:, :, 0] = True  # Top boundary
        boundary_boolarray[:, :, -1] = True  # Bottom boundary

    else:
        raise ValueError("Array dimension not supported. Only 2D and 3D arrays are supported.")

    return boundary_boolarray


def detect_islands_with_periodicity(array, struc=struc_1_3D):
    """
    Function to detect "islands" as material inclusions in pore are not possible

    Parameters
    ----------
    array : voxel array that is filled with 1 (e.g. material) or 0 (e.g. pore)
    stuc:   structure that determines what accounts for the connectivity

    Returns
    -------
    array : boolean array with True values for island voxels

    """
    num_dims = array.ndim

    # get all islands of the array
    ##############################

    # get labeled array (all islands of materials have different numbers except pores)
    labeled_array, num_features = label(array, structure=struc)
    # delete zeros from labeled array (do not need to be changed in postprocessing)
    labeled_array_without_0 = np.delete(labeled_array.ravel(), np.where(labeled_array.ravel() == 0))
    # get unique labels and count them
    unique, counts = np.unique(labeled_array_without_0, return_counts=True)
    # get largest connected area (do not need to be changed in postprocessing)
    connected_area = list(np.where(counts == max(counts))[0])

    # get island labels without label for largest connected area
    island_labels_mask = np.in1d(unique, unique[connected_area], invert=True)
    island_labels = unique[island_labels_mask]

    # set boolean array entries to True for islands
    array_islands = np.isin(labeled_array, island_labels)

    # check on islands if they are connected with the largest connected area
    # due to periodicity
    ########################################################################

    # mark all boundary voxels with True
    boundary_array = mark_boundary_voxels(array)
    # combine island voxel and boundary voxels -> boolean array
    combined_array = boundary_array & array_islands
    # find the indices of true values in combined arrays
    combined_indices = np.argwhere(combined_array)

    # find the indices of the opposite sides of true values
    indices_opposite = []
    for idx_number, idx in enumerate(combined_indices):

        if num_dims == 2:
            if idx[0] == 0 or idx[0] == combined_array.shape[0] - 1:
                append_idx = [
                    combined_array.shape[0] - 1 - combined_indices[idx_number][0],
                    combined_indices[idx_number][1],
                ]
            elif idx[1] == 0 or idx[1] == combined_array.shape[1] - 1:
                append_idx = [
                    combined_indices[idx_number][0],
                    combined_array.shape[1] - 1 - combined_indices[idx_number][1],
                ]
            else:
                pass

        elif num_dims == 3:
            if idx[0] == 0 or idx[0] == combined_array.shape[0] - 1:
                append_idx = [
                    combined_array.shape[0] - 1 - combined_indices[idx_number][0],
                    combined_indices[idx_number][1],
                    combined_indices[idx_number][2],
                ]
            elif idx[1] == 0 or idx[1] == combined_array.shape[1] - 1:
                append_idx = [
                    combined_indices[idx_number][0],
                    combined_array.shape[1] - 1 - combined_indices[idx_number][1],
                    combined_indices[idx_number][2],
                ]
            elif idx[2] == 0 or idx[2] == combined_array.shape[2] - 1:
                append_idx = [
                    combined_indices[idx_number][0],
                    combined_indices[idx_number][1],
                    combined_array.shape[1] - 1 - combined_indices[idx_number][2],
                ]
            else:
                pass

        else:
            raise ValueError("Array dimension not supported. Only 2D and 3D arrays are supported.")
        indices_opposite.append(append_idx)

    array_opposite = np.array(indices_opposite)

    # check if at the opposite side is material -> update the whole island
    no_islands = []
    for idx_num, idx_check in enumerate(array_opposite):

        if num_dims == 2:
            island_number = labeled_array[idx_check[0], idx_check[1]]

            if island_number in (unique[connected_area]):
                no_island_number = labeled_array[combined_indices[idx_num][0], combined_indices[idx_num][1]]
                no_islands.append(no_island_number)

            else:
                pass

        elif num_dims == 3:
            island_number = labeled_array[idx_check[0], idx_check[1], idx_check[2]]

            if island_number in (unique[connected_area]):
                no_island_number = labeled_array[
                    combined_indices[idx_num][0], combined_indices[idx_num][1], combined_indices[idx_num][2]
                ]
                no_islands.append(no_island_number)
            else:
                pass

    # update array_island (analogously to steps on top)
    no_islands_unique = list(np.unique(no_islands))
    for i in no_islands_unique:
        connected_area.append(int(np.where(unique == i)[0]))
    island_labels_mask_update = np.in1d(unique, unique[connected_area], invert=True)
    island_labels_update = unique[island_labels_mask_update]
    array_islands = np.isin(labeled_array, island_labels_update)

    return array_islands


if __name__ == "__main__":
    # examples of using the detect_islands_with_periodicity function
    ################################################################

    a = np.array(
        [
            [0, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 1, 1, 1, 0],
        ]
    )

    b1 = np.array(
        [
            [[1, 1, 1], [0, 1, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 0, 1], [0, 0, 0], [0, 1, 1], [0, 0, 0]],
            [[0, 0, 1], [0, 0, 0], [0, 1, 1], [0, 0, 0]],
        ]
    )

    b2 = np.array(
        [
            [[1, 1, 1], [0, 1, 0], [1, 0, 1], [1, 1, 1]],
            [[0, 0, 0], [1, 1, 1], [0, 1, 0], [0, 0, 0]],
            [[1, 0, 1], [0, 0, 0], [0, 0, 0], [1, 0, 1]],
            [[1, 1, 1], [0, 1, 0], [1, 0, 1], [1, 1, 1]],
        ]
    )

    # c = np.load('RVE_100_3_reconstructed.npy')
    # d = np.load('LIST_20230331/last_frame_l100.npy')
    c = np.load("C:\\Users\\Seibert\\Documents\\Code\\mcrpy\\microstructures\\RVE_100_3.npy")
    d = np.load("C:\\Users\\Seibert\\Documents\\Code\\mcrpy\\microstructures\\lang_100_rec.npy")
    e = np.load("C:\\Users\\Seibert\\Documents\\Code\\mcrpy\\mcrpy\\mcrpy\\results\\last_frame.npy")

    # depends on structure element: struc_1 gives an other output than struc_2
    array_boolean_test_a = detect_islands_with_periodicity(a, struc_2_2D)
    array_boolean_test_b1 = detect_islands_with_periodicity(b1, struc_2_3D)
    array_boolean_test_b2 = detect_islands_with_periodicity(b2, struc_2_3D)
    array_boolean_test_c = detect_islands_with_periodicity(c, struc_2_3D)
    print(np.sum(array_boolean_test_c))
    array_boolean_test_d = detect_islands_with_periodicity(d, struc_2_3D)
    print(np.sum(array_boolean_test_d))
    array_boolean_test_c1 = detect_islands(c)
    print(np.sum(array_boolean_test_c1))
    array_boolean_test_d1 = detect_islands(d)
    print(np.sum(array_boolean_test_d1))

    array_boolean_test_e = detect_islands_with_periodicity(e, struc_1_3D)
    print(np.sum(array_boolean_test_e))
