"""
Functions for the calculation of the geometrical tortuosity
Method: Skeleton shortest path searching method (SSPSM)

First idea - Reference: Al-Raoush, R. I., & Madhoun, I. T. (2017). TORT3D: A MATLAB code to compute geometric tortuosity from 3D images of unconsolidated porous media. Powder Technology, 320, 99-107.
"""

from __future__ import annotations

import tensorflow as tf

from mcrpy.src import descriptor_factory
from mcrpy.descriptors.PhaseDescriptor3D import PhaseDescriptor3D

import numpy as np

# from scipy.io import loadmat
from skimage.morphology import skeletonize  # , skeletonize_3d, medial_axis
import matplotlib.pyplot as plt

# from Create3Darray import create3Darray
from matplotlib.animation import PillowWriter
import matplotlib.animation as ani

# from numba import njit


class Tort3D(PhaseDescriptor3D):
    """3D Tortuosity, see module docstring. """

    is_differentiable = False

    @staticmethod
    def make_singlephase_descriptor(**kwargs) -> callable:

        # @tf.function
        def compute_descriptor(microstructure: np.ndarray) -> np.ndarray:
            microstructure = np.round(microstructure, decimals=0).astype(int)
            assert not np.any(np.isnan(microstructure))
            if microstructure.shape[0] == 1:
                microstructure = microstructure[0]
            if microstructure.shape[-1] == 1:
                microstructure = microstructure[..., 0]
            desc = []
            arr = 1 - microstructure
            for direction in range(3):
                desc.append(main_tort(arr, direction, _connection_type=26))
            return np.stack(desc)

        return compute_descriptor


def register() -> None:
    descriptor_factory.register("Tort3D", Tort3D)


def plot_fig(_image: np.ndarray, _filename: str, _path_to_file: str = "") -> None:
    """
    A general function to plot figures with the same settings
    """

    plt.figure(1, figsize=(10, 10), dpi=144)
    plt.imshow(_image, cmap="bwr", vmin=0, vmax=3)
    plt.axis("off")
    plt.savefig(_path_to_file + _filename + ".png")


def skeletonize3D(
    _image: np.ndarray,
    _direction_flow: int,
    _starting_depth: int = 0,
    _plot_skel: bool = False,
    _fname: bool = None,
    _DIRname: bool = None,
):
    """
    Function to skeletonize the voxel-based image according to SSPSM


    _image :            voxel-based image
    _direction_flow :   tortuosity in a specific direction
    _starting_depth :   Starting depth of finding starting points (default: 0)
    _plot_skel :        Plot and save skeleton figure (default: False)
    _DIRname:           Name of Directory
    _fname:             Filename


    medial_axis_image :             voxel-based image with highlighted skeleton
    nrows :                         number of rows
    ncolumns :                      number of columns
    ndepth :                        number of "layers"
    cum_number_medial_each_slide :  number of skeleton voxels for each slide
    skeleton_volume_axis_index :    indices of skeleton voxels
    """

    # first: change order of image array for the respective direction
    if _direction_flow == 0:  # order after this: [y, z, x]
        _image = _image.transpose(1, 2, 0)
    elif _direction_flow == 1:  # order after this: [z, x, y]
        _image = _image.transpose(2, 0, 1)
    elif _direction_flow == 2:  # order after this: [x, y, z]
        pass
    else:
        raise ValueError("Flow direction has to be either 0 (x), 1 (y) or 2 (z).")

    # dimensions of image
    nrows = _image.shape[0]  # number of rows (x)
    ncolumns = _image.shape[1]  # number of columns (y)
    ndepth = _image.shape[2]  # number of slices in z (z)

    # create empty lists
    skeleton_volume_axis_index = []
    number_medial_each_slide = []
    medial_axis_location_all = []

    # medial axis image generation
    # imagse_orig = np.copy(image)
    medial_axis_image = _image

    # skeletonization
    for depth in range(ndepth):

        binary_slice = _image[:, :, depth]  # Extract the 2D slice
        binary_slice = binary_slice.astype(bool).copy(
            order="C"
        )  # Ensure the binary slice is in the correct format (dtype=bool) and C-contiguous
        medial_skeleton_slice = skeletonize(1 - binary_slice)  # Perform skeletonization on the binary slice

        medial_axis_location = np.where(
            medial_skeleton_slice == True
        )  # find all values in array ==1 (list of index; looks at first coloumn, then in second coloumn etc.)
        medial_axis_indices_slice = np.sort(
            medial_axis_location[0] + nrows * medial_axis_location[1]
        )  # local index; looks at first coloumn, then in second coloumn etc.

        medial_axis_location_all.append(medial_axis_location)
        number_medial_each_slide.append(len(medial_axis_indices_slice))
        skeleton_volume_axis_index.extend(
            medial_axis_indices_slice + nrows * ncolumns * depth
        )  # Append the skeletonized slice to the result volume

        for coor in range(len(medial_axis_location[0])):
            x_coor = medial_axis_location[0][coor]
            y_coor = medial_axis_location[1][coor]
            medial_axis_image[x_coor, y_coor, depth] = 2

    # Convert the list of skeletonized slices back to a 3D array
    skeleton_volume_axis_index = np.array(skeleton_volume_axis_index)
    cum_number_medial_each_slide = np.cumsum(number_medial_each_slide)  # number of skeleton voxels per slide summed up
    number_medial_each_slide = np.insert(number_medial_each_slide, 0, [0, 0])  # adds 1,1 before vector
    cum_number_medial_each_slide = np.insert(cum_number_medial_each_slide, 0, [0, 0])  # adds 1,1 before vector

    if _plot_skel:
        # plot of the skeleton of the first slide
        skeleton_first_slide = medial_axis_image[:, :, 0]
        plot_fig(skeleton_first_slide, "Skeleton_first_slide_" + _fname, _DIRname)

    return (
        medial_axis_image,
        nrows,
        ncolumns,
        ndepth,
        cum_number_medial_each_slide,
        number_medial_each_slide,
        skeleton_volume_axis_index,
    )


# @njit
def find_junctions(_image_orig: np.ndarray, ncolumns: int, nrows: int, ndepth: int):
    """
    Function to find the junctions of the skeleton on the first slide
    -> these are handled as starting points for the running path function

    _medial_axis_image :    voxel-based image with highlighted skeleton
    nrows :                 number of rows
    ncolumns :              number of columns
    ndepth :                number of "layers"

    junctions_index : indices of junctions at the first slide
    """

    # get the first skeleton slice (again)
    first_slide = (np.copy(_image_orig[:, :, 0])).astype(bool).copy(order="C")  # Extract the 2D slice
    # first_slide = first_slide.astype(bool).copy(order='C')    # Ensure the binary slice is in the correct format (dtype=bool) and C-contiguous
    first_skeleton_slice = skeletonize(1 - first_slide)  # Perform skeletonization on the binary slice
    # Define the 8-connectivity neighbors
    neighbors = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    junctions = []
    for i in range(1, first_skeleton_slice.shape[0] - 1):
        for j in range(1, first_skeleton_slice.shape[1] - 1):
            if first_skeleton_slice[i, j] == 1:
                count = 0
                for dx, dy in neighbors:
                    if first_skeleton_slice[i + dx, j + dy] == 1:
                        count += 1
                if count > 2 or count == 0:
                    junctions.append((i, j))
    if len(junctions) == 0:
        np.save("zero_junctions_found.npy", _image_orig)
        raise ValueError
    junctions = np.transpose(np.array([np.array(junctions)[:, 0], np.array(junctions)[:, 1], np.zeros(len(junctions))]))
    junctions_index = junctions[:, 0] + nrows * junctions[:, 1] + nrows * ncolumns * junctions[:, 2]

    return junctions_index


# @njit
def get_pixels_within_radius(center_x: int, center_y: int, radius: int):
    """
    Function to get all pixels for a circle with a certain radius


    center_x : center of circle (x coordinate, integer)
    center_y : center of circle (y coordinate, integer)
    radius : radius of circle (integer)


    pixels_within_radius : all pixels within the circle (array)
    """

    pixels_within_radius = []

    for i in range(center_x - radius, center_x + radius + 1):
        for j in range(center_y - radius, center_y + radius + 1):
            distance = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
            if distance <= radius:
                pixels_within_radius.append((i, j))

    pixels_within_radius = np.array(pixels_within_radius)

    return pixels_within_radius


# @njit
def find_starting_points(
    _image: np.ndarray,
    _junctions_index: np.ndarray,
    _starting_depth: int,
    nrows: int,
    ncolumns: int,
    _reduce_SP: float = 1.0,
    _plot_SP: bool = False,
    _fname: str = None,
    _DIRname: str = None,
):
    """
    Function to find the starting points.
    It removes "junctions" that are very close to each other.


    _image :            voxel-based image (array)
    _junctions_index :  indices of junctions at the first slide (array)
    _reduce_SP :        reduction of number of starting points for less computational effort
                        [between 0.0 (0 starting points) and 1.0 (all starting points)] (default: 1.0, float)
    _starting_depth :   Starting depth of finding starting points (default: 0; integer)
    nrows :             number of rows
    ncolumns :          number of columns
    _plot_SP :          Plot and save figure with starting points with circles(default: False; Boolean)
    _DIRname:           Name of Directory
    _fname:             Filename


    SP_loc_rad_shuffle : starting point locations with radius of circles
    """
    # Starting point indices
    starting_points_index = _junctions_index  # actually connect_voxel_index_filtered2

    # starting points positions of index that are == 2 at the starting slide
    z_SP = np.floor((starting_points_index) / (nrows * ncolumns))
    y_SP = np.floor(starting_points_index / nrows) - (ncolumns * (z_SP))
    x_SP = starting_points_index - nrows * (y_SP) - nrows * ncolumns * (z_SP)
    location_SP = np.transpose(np.array([x_SP, y_SP, z_SP]))  # starting point positions

    # Final locations of starting points
    ####################################

    slide_one = _image[:, :, _starting_depth]
    slide_one_copy = np.copy(slide_one)
    slide_one_copy2 = np.copy(slide_one)

    all_radius = []

    for m in range(len(starting_points_index)):

        # print(f"{m+1} of {len(starting_points_index)} starting points")
        radius = 0  # set radius to zero
        # circle_points_index= starting_points_index[m]    # circle point index (Index of starting point)
        ypos_circle = int(location_SP[m][1])
        xpos_circle = int(location_SP[m][0])
        ypos_center = int(location_SP[m][1])
        xpos_center = int(location_SP[m][0])

        temp_positions_x = []  # temporary file for positions in x direction
        temp_positions_y = []  # temporary file for positions in y direction

        while np.count_nonzero(slide_one[xpos_circle, ypos_circle] == 1) < 1:  # counts

            pixels_within_radius = get_pixels_within_radius(xpos_center, ypos_center, radius)

            condition1 = pixels_within_radius[:, 0] < nrows
            condition2 = pixels_within_radius[:, 1] < ncolumns
            condition3 = (pixels_within_radius >= 0).all(axis=1)
            combined_conditions = condition1 & condition2 & condition3

            pixels_within_radius_filtered = pixels_within_radius[combined_conditions]

            xpos_circle = pixels_within_radius_filtered[:, 0]
            ypos_circle = pixels_within_radius_filtered[:, 1]

            temp_positions_x.append(xpos_circle)
            temp_positions_y.append(ypos_circle)

            radius += 1

        # print("Material was hit ...")
        slide_one[temp_positions_x[-2], temp_positions_y[-2]] = 3  # the second last entry is adopted
        slide_one_copy[:] = 0
        all_radius.append(radius - 2)

    # summary of starting points, starting point location and radius
    SP_loc_rad = np.hstack((location_SP, np.array(all_radius).reshape(-1, 1)))
    temp_SP_loc_rad = np.copy(SP_loc_rad)  # temporary file

    # remove starting points when circles overlap
    removed_SP = []
    entry_ind = 0
    count_new_radius = 0

    while len(temp_SP_loc_rad) >= 2:

        # sorted_indices = np.argsort(temp_SP_loc_rad[:,-1])#[::-1]
        # temp_SP_loc_rad = temp_SP_loc_rad[sorted_indices]
        current_entry = temp_SP_loc_rad[0]

        # delete entry from temporary file
        temp_SP_loc_rad = temp_SP_loc_rad[1:]

        # compute the distance between temporary entry and all other starting
        # Points and substract the radius of both circles
        remove_dist = np.sqrt(
            ((current_entry[0] - temp_SP_loc_rad[:, 0]) ** 2)
            + ((current_entry[1] - temp_SP_loc_rad[:, 1]) ** 2)
            + ((current_entry[2] - temp_SP_loc_rad[:, 2]) ** 2)
        ) - (current_entry[3] + temp_SP_loc_rad[:, 3])

        # see whether the circle overlap with any other circle
        changing_indices = np.where(remove_dist < 0)
        # temp_changing_SP = temp_SP_loc_rad[changing_indices]

        # compute new radius in order to take entry as a starting point too
        new_radius_ce = current_entry[3] + remove_dist[changing_indices] / 2  # current entry
        new_radius_ci = temp_SP_loc_rad[changing_indices][:, 3] + remove_dist[changing_indices] / 2  # changing indices

        if new_radius_ce.size > 0:
            if np.any(new_radius_ce < 1) or np.any(new_radius_ci < 1):
                # if any radius <= 1: append current starting point to removed list
                removed_SP.append(current_entry)

            else:
                # change the radius of the current entry and of the other entries
                SP_loc_rad[entry_ind][3] = np.floor(np.min(new_radius_ce))

                tmp = temp_SP_loc_rad[changing_indices]
                tmp[:, 3] = np.floor(new_radius_ci)

                temp_SP_loc_rad[changing_indices] = tmp

                for row_temp in temp_SP_loc_rad[changing_indices]:
                    find_row_SP = np.where(np.all(row_temp[0:3] == SP_loc_rad[:, 0:3], axis=1))
                    SP_loc_rad[find_row_SP, 3] = row_temp[3]

                count_new_radius += 1

        entry_ind += 1

    # print("Counter of setting a new radius:", count_new_radius)
    # delete removed locations from raw_location_radius and get the final
    # starting points
    removed_SP_filtered = [arr for arr in removed_SP if arr.size > 0]
    if removed_SP_filtered:
        removing_locations = np.vstack(removed_SP_filtered)

        # remove locations from initial array
        # Compare every row of array1 to every row in array2
        is_inside_list = []
        for row_ri in removing_locations:
            is_inside = np.where(np.all(row_ri == SP_loc_rad, axis=1))
            is_inside_list.extend(is_inside)

        is_inside_list = [arr for arr in is_inside_list if arr.size > 0]
        keep_rows = np.ones(SP_loc_rad.shape[0], dtype=bool)
        keep_rows[np.vstack(is_inside_list)] = False

        # list of starting points with radius of circle
        SP_loc_rad = SP_loc_rad[keep_rows]

    # final starting points
    #######################

    # reduction of number of starting points
    number_reduction_SP = int(np.round(len(SP_loc_rad) * _reduce_SP))
    random_entries_SP = np.random.choice(SP_loc_rad.shape[0], size=number_reduction_SP, replace=False)

    SP_loc_rad_shuffle = SP_loc_rad[random_entries_SP, :]

    # Mark final starting points with circles ...
    ############################################

    if _plot_SP:
        circles = np.transpose(np.array([SP_loc_rad_shuffle[:, 0], SP_loc_rad_shuffle[:, 1], SP_loc_rad_shuffle[:, 3]]))

        def mark_pixels_within_circles(array: np.ndarray, circles: np.ndarray):
            """
            Function to mark the first slide of the microstructure with circles
            -> Starting points for searching the paths

            """
            x, y = np.meshgrid(np.arange(array.shape[0]), np.arange(array.shape[1]), indexing="ij")

            for center_x, center_y, radius in circles:
                distances = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                array[(distances <= radius)] = 3

        mark_pixels_within_circles(slide_one_copy2, circles)

        # Visualize the medial axis image and the starting points for the first slice
        plot_fig(slide_one_copy2, "Skeleton_Circles_SP_" + _fname, _DIRname)

    return SP_loc_rad_shuffle


def running_paths(
    _image: np.ndarray,
    _SP_loc_rad_shuffle: np.ndarray,
    nrows: int,
    ncolumns: int,
    ndepth: int,
    cum_number_medial_each_slide: np.ndarray,
    number_medial_each_slide: np.ndarray,
    skeleton_volume_axis_index: np.ndarray,
    _connect_search: int = 26,
    _plot: bool = False,
    _DIRname: str = None,
    _fname: str = None,
):
    """
    Function for running path
    -> for each starting point: the fastest way through the microstructure is searched


    _image :                        voxel-based image
    _SP_loc_rad_shuffle :           starting point locations with radius of circles
    nrows :                         number of rows
    ncolumns :                      number of columns
    ndepth :                        number of "layers"
    cum_number_medial_each_slide :  number of skeleton voxels for each slide
    number_medial_each_slide :      number of skeleton voxels per slide
    skeleton_volume_axis_index :    indices of skeleton voxels
    _connect_search :               Type of voxel connection; Different values are possible (3D):
                                    6:  Connection via faces
                                    18: Connection via faces and edges
                                    26: Connection via faces, edges and points
                                    Default: 26
    _plot:                          Plot of the individual paths (default: False)
    _DIRname:                       Name of Directory
    _fname:                         Filename


    all_paths_length : path lengths of all paths through the microstructure (list)
    """

    non_connected_voxels = 0
    keep_medial_axis = []  # create keep medial axis
    all_paths_length = []  # path lengths

    for path in range(len(_SP_loc_rad_shuffle)):

        # print(f"Path number {path+1} of {len(_SP_loc_rad_shuffle)}")

        # location of current starting point
        next_move_location = _SP_loc_rad_shuffle[path, 0:3]
        next_move_index = (
            next_move_location[0] + nrows * next_move_location[1] + nrows * ncolumns * next_move_location[2]
        )

        # create empty lists
        path_index = []
        path_location = []
        path_location.append(next_move_location)
        path_index.append(next_move_index)

        # set global path corrector to One
        path_corrector = 1

        while (next_move_location[2] < (ndepth - 1)) and path_corrector == 1:  # len(image[0]) - 1)

            # get all neigbouring voxel indexes
            connect_voxel_coor = get_neighboring_voxels(nrows, ncolumns, ndepth, next_move_location, _connect_search)
            connect_voxel_index = (
                connect_voxel_coor[:, 0]
                + nrows * connect_voxel_coor[:, 1]
                + nrows * ncolumns * connect_voxel_coor[:, 2]
            )

            # find the number of skeleton voxels of the previous, the current and the next slide
            x1 = cum_number_medial_each_slide[int(next_move_location[2])]
            x2 = cum_number_medial_each_slide[int(next_move_location[2]) + 3] + 1

            # find common set of neighboring voxels and voxels that belong to the skeleton surface
            next_move_match = np.isin(
                connect_voxel_index, skeleton_volume_axis_index[x1:x2]
            )  # boolean array with matching voxels
            connect_voxel_index = connect_voxel_index[next_move_match]  # reduce it to the neighboring skeleton voxels

            # find common set of neighboring cells (==skeleton) and path index and delete the previous path index
            connect_path_match = np.isin(connect_voxel_index, path_index)
            connect_voxel_index = connect_voxel_index[np.invert(connect_path_match)]

            current_match = np.isin(connect_voxel_index, next_move_index)
            connect_voxel_index = connect_voxel_index[np.invert(current_match)]

            if len(connect_voxel_index) > 0:

                # get location of indexes
                # starting points positions of index that are == 2 at the starting slide
                z_index = np.floor((connect_voxel_index) / (nrows * ncolumns))
                y_index = np.floor(connect_voxel_index / nrows) - (ncolumns * (z_index))
                x_index = connect_voxel_index - nrows * (y_index) - nrows * ncolumns * (z_index)

                location_coord = np.transpose(np.array([x_index, y_index, z_index]))  # starting point positions

                # find next moving location
                next_move_location = location_coord[np.where(location_coord[:, 2] == np.max(location_coord[:, 2]))[0]]
                next_move_location_choice = np.random.randint(0, len(next_move_location))
                next_move_location = next_move_location[next_move_location_choice]

                # find next moving index
                next_move_index = (
                    next_move_location[0] + nrows * next_move_location[1] + nrows * ncolumns * next_move_location[2]
                )

                # append to pathes
                path_index.append(next_move_index)
                path_location.append(next_move_location)

            elif len(path_location) > 1:

                # loads in last path location and index
                next_move_location = path_location[-2]
                next_move_index = path_index[-2]

                # remove index from skeleton axis
                remove_loc = np.where(skeleton_volume_axis_index == path_index[-1])
                temp_keep_medial_axis = skeleton_volume_axis_index[remove_loc]
                keep_medial_axis.append(temp_keep_medial_axis)
                skeleton_volume_axis_index = np.delete(skeleton_volume_axis_index, remove_loc)

                # update number of medial axis of each slide and summed number of skeleton voxels per slide
                number_medial_each_slide[int(next_move_location[2]) + 2] = (
                    number_medial_each_slide[int(next_move_location[2]) + 2] - 1
                )
                cum_number_medial_each_slide = np.cumsum(number_medial_each_slide)

                # update; removes last entry from path way
                path_index = path_index[:-1]
                path_location = path_location[:-1]

            else:
                # print(f"Path with path number {path+1} is a non-connected path")
                path_corrector = 0
                non_connected_voxels += 1

        if len(path_index) > 1 and path_corrector != 0:

            all_paths_length.append(len(path_index))  # save path length
            path_location = (np.array(path_location)).astype(int)  # convert list to array
            _image[path_location[:, 0], path_location[:, 1], path_location[:, 2]] = (
                4 + path
            )  # mark path with specific number

    all_paths_length = [
        length for length in all_paths_length if length > 1
    ]  # delete all "paths" that have the length "1"

    if _plot:
        plot_paths(_image, _DIRname, "Paths_" + _fname)

    return all_paths_length


def statistical_tort(
    _all_paths_length: list,
    nrows: int,
    _plot: bool = False,
    _SP_loc_rad: np.ndarray = None,
    _fname: str = None,
    _DIRname: str = None,
):
    """
    Function to calculate different statistical tortuosity values


    _all_paths_length : Path lengths of all paths through the microstructure (list)
    nrows :             Number of rows
    _plot :             Plot and save figure with statistical distribution of tortuosity (default: False; Boolean)
    _SP_loc_rad:        Array with starting point locations and radius (only necessary for plot)
    _DIRname:           Name of directory
    _fname:             Filename


    tortuosity_mean : mean tortuosity
    tortuosity_std : standard deviation of tortuosity
    tortuosity_median : median of tortuosity
    """

    tortuosity_distri = np.array(_all_paths_length) / nrows  # tortuosity value for each path
    tortuosity_mean = np.mean(tortuosity_distri)  # mean tortuosity
    tortuosity_std = np.std(tortuosity_distri)  # standard derivation of tortuosity path
    tortuosity_median = np.median(tortuosity_distri)  # median value of tortuosity

    if _plot:
        plot_tort_distribution(_SP_loc_rad, tortuosity_distri, _DIRname, "Tort_Distribution_" + _fname)

    return tortuosity_mean, tortuosity_std, tortuosity_median


# additional functions for plots
################################


def plot_paths(image: np.ndarray, path_to_file: str, _filename: str, remove_voxels: str = True):

    image = image.astype(float)

    if remove_voxels:
        image[image < 4] = np.nan  # <3 shows also the circles of the starting points

    # Create a figure and a 3D subplot
    fig = plt.figure(num=100, figsize=(10, 10), dpi=144)
    ax = fig.add_subplot(111, projection="3d")

    # view
    ax.view_init(elev=20, azim=-80)  # default: 30, -60

    cmap = plt.get_cmap("rainbow")

    if remove_voxels:
        norm = plt.Normalize(4, np.unique(image)[-2])
    else:
        norm = plt.Normalize(0, 1)
    mask = ~np.isnan(image)
    # Map data values to colors
    colors = cmap(norm(image))
    ax.voxels(mask, facecolors=colors)

    # Remove the grid of the coordinate axis
    ax.grid(False)

    # Title
    # plt.title("Paths")

    # Hide the axis numbers
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # save fig
    plt.savefig(path_to_file + _filename + ".png")

    # Show the plot
    plt.show()


def plot_tort_distribution(SP_loc_rad_shuffle: np.ndarray, tort_values: np.ndarray, path_to_file: str, _filename: str):
    tortu_min = np.min(tort_values)
    tortu_max = np.max(tort_values)

    segments = np.linspace(tortu_min, tortu_max, 20)

    segment_numbers = []
    for i in range(len(segments) - 1):
        number_of_values = np.count_nonzero((tort_values >= segments[i]) & (tort_values <= segments[i + 1]))
        segment_numbers.append(number_of_values)

    probability = np.array(segment_numbers) / len(SP_loc_rad_shuffle)

    segments_avg = (segments[:-1] + segments[1:]) / 2

    segments = np.round(segments, 4)
    plt.figure(num=101, figsize=(10, 8), dpi=200)
    plt.bar(segments_avg, probability * 100, width=0.045)
    plt.xticks(
        segments,
        [
            f"{segments[0]}",
            "",
            "",
            "",
            f"{segments[4]}",
            "",
            "",
            "",
            "",
            f"{segments[9]}",
            "",
            "",
            "",
            "",
            f"{segments[14]}",
            "",
            "",
            "",
            "",
            f"{segments[19]}",
        ],
    )
    plt.xlabel(r"Tortuosity $\tau$")
    plt.ylabel("Probability in $\%$")

    plt.savefig(path_to_file + _filename + ".png")

    plt.show()


def Animate_Tort_path(*path_histories_and_color, image: np.ndarray, _filename: str):
    """
    Function creates an animation so that the random walk can be seen
    ax.scatter works much faster than ax.voxels (but not as wonderful :D )
    """

    def ani_func(i=int):

        # Clear the previous scatter plot
        for scatter in ax.collections:
            scatter.remove()

        for path_history, color in path_histories_and_color:
            if i < len(path_history):
                plot_indices = path_history[i]
                ax.scatter(plot_indices[:, 0], plot_indices[:, 1], plot_indices[:, 2], marker="s", c=color)

    fig = plt.figure(num=102, figsize=(20, 20), dpi=200)
    fig.patch.set_facecolor("none")  # Set the figure background to be transparent
    ax = fig.add_subplot(111, projection="3d")

    # view
    ax.view_init(elev=20, azim=-80)  # default: 30, -60

    # darker shade of grey
    ax.set_facecolor((1, 1, 1))  # 1: white

    # Remove the grid of the coordinate axis
    ax.grid(False)

    # axis limits
    ax.set_xlim([0, image.shape[0]])
    ax.set_ylim([0, image.shape[1]])
    ax.set_zlim([0, image.shape[2]])

    # Hide the axis numbers
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Wenn _phi zu lang ist, lieber einen kleineren Wert fuer frames setzen,
    # z.B. franes = 150
    frames = max(len(path_history) for path_history, _ in path_histories_and_color)
    interval = 200  # milliseconds (does not make any difference yet)
    animator = ani.FuncAnimation(fig, ani_func, frames=frames, interval=interval)
    # animator.save("../Ergebnisse_Abaqus/Diagramme/"+_filename+'_RM3_phi3.gif')
    # Set up formatting for the movie files
    writer = PillowWriter(fps=15, metadata=dict(artist="E.Langner"), bitrate=-1)  # , bitrate=1800
    animator.save("Animationen/" + _filename + ".gif", writer=writer)
    plt.show()


"""
Function to generate a the neighboring cells
"""
import numpy as np


def get_neighboring_voxels(nrows, ncolumns, ndepth, coord, connect):
    """
    Function to generate a the neighboring cells
    In 3D: maximum number of connected cells: 26

    Parameters
    ----------
    nrows :     number of rows
    ncolumns :  number of columns
    ndepth :    number of depth
    coord :     Coordinates of the medial axis (skeleton) indicator
    connect :   Type of connection
                2D: 4, 8; in 3D: 6, 18, 26

    Returns
    -------
    connect_voxels_coordinates : coordinates of the connected voxels
    """

    def neighbour(coord, nx=0, ny=0, nz=0):

        # is the neighbor of the *previous/current/next slide* with on the *top/same/bottom height and left/middle/right position*
        if coord.ndim > 1:
            neigbour = np.transpose(np.array([coord[:, 0] + nx, coord[:, 1] + ny, coord[:, 2] + nz]))
            # search for unrealistic results and change row to initial coordinate
            false_rows_1 = np.any((neigbour < 0), axis=1)  # are there any negative values?
            false_rows_2 = (
                (neigbour[:, 0] >= nrows) | (neigbour[:, 1] >= ncolumns) | (neigbour[:, 2] >= ndepth)
            )  # are there any values > size?
        else:
            neigbour = np.transpose(np.array([coord[0] + nx, coord[1] + ny, coord[2] + nz]))
            # search for unrealistic results and change row to initial coordinate
            false_rows_1 = np.any((neigbour < 0))  # are there any negative values?
            false_rows_2 = (
                (neigbour[0] >= nrows) | (neigbour[1] >= ncolumns) | (neigbour[2] >= ndepth)
            )  # are there any values > size?

        # Modify the neighboring cells independently for each row
        if np.any(false_rows_1):
            coord_subset = coord[false_rows_1]
            neigbour[false_rows_1] = coord_subset
        if np.any(false_rows_2):
            coord_subset = coord[false_rows_2]
            neigbour[false_rows_2] = coord_subset

        return neigbour

    n_1 = neighbour(
        coord, nx=-1, ny=-1, nz=-1
    )  # neighbor of the *previous slide* with on the *top height and left position*
    n_2 = neighbour(
        coord, nx=-1, ny=0, nz=-1
    )  # neighbor of the *previous slide* with on the *same height and left position*
    n_3 = neighbour(
        coord, nx=-1, ny=+1, nz=-1
    )  # neighbor of the *previous slide* with on the *bottom height and left position*
    n_4 = neighbour(
        coord, nx=0, ny=-1, nz=-1
    )  # neighbor of the *previous slide* with on the *top height and mddle position*
    n_5 = neighbour(
        coord, nx=0, ny=0, nz=-1
    )  # neighbor of the *previous slide* with on the *same height and mddle position*
    n_6 = neighbour(
        coord, nx=0, ny=+1, nz=-1
    )  # neighbor of the *previous slide* with on the *bottom height and mddle position*
    n_7 = neighbour(
        coord, nx=+1, ny=-1, nz=-1
    )  # neighbor of the *previous slide* with on the *top height and right position*
    n_8 = neighbour(
        coord, nx=+1, ny=0, nz=-1
    )  # neighbor of the *previous slide* with on the *same height and right position*
    n_9 = neighbour(
        coord, nx=+1, ny=+1, nz=-1
    )  # neighbor of the *previous slide* with on the *bottom height and right position*
    n_10 = neighbour(
        coord, nx=-1, ny=-1, nz=0
    )  # neighbor of the *current slide* with on the *top height and left position*
    n_11 = neighbour(
        coord, nx=-1, ny=0, nz=0
    )  # neighbor of the *current slide* with on the *same height and left position*
    n_12 = neighbour(
        coord, nx=-1, ny=+1, nz=0
    )  # neighbor of the *current slide* with on the *bottom height and left position*
    n_13 = neighbour(
        coord, nx=0, ny=-1, nz=0
    )  # neighbor of the *current slide* with on the *top height and middle position*
    n_14 = neighbour(
        coord, nx=0, ny=+1, nz=0
    )  # neighbor of the *current slide* with on the *bottom height and middle position*
    n_15 = neighbour(
        coord, nx=+1, ny=-1, nz=0
    )  # neighbor of the *current slide* with on the *top height and right position*
    n_16 = neighbour(
        coord, nx=+1, ny=0, nz=0
    )  # neighbor of the *current slide* with on the *same height and right position*
    n_17 = neighbour(
        coord, nx=+1, ny=+1, nz=0
    )  # neighbor of the *current slide* with on the *bottom height and right position*
    n_18 = neighbour(
        coord, nx=-1, ny=-1, nz=+1
    )  # neighbor of the *next slide* with on the *top height and left position*
    n_19 = neighbour(
        coord, nx=-1, ny=0, nz=+1
    )  # neighbor of the *next slide* with on the *same height and left position*
    n_20 = neighbour(
        coord, nx=-1, ny=+1, nz=+1
    )  # neighbor of the *next slide* with on the *bottom height and left position*
    n_21 = neighbour(
        coord, nx=0, ny=-1, nz=+1
    )  # neighbor of the *next slide* with on the *top height and middle position*
    n_22 = neighbour(
        coord, nx=0, ny=0, nz=+1
    )  # neighbor of the *next slide* with on the *same height and middle position*
    n_23 = neighbour(
        coord, nx=0, ny=+1, nz=+1
    )  # neighbor of the *next slide* with on the *bottom height and middle position*
    n_24 = neighbour(
        coord, nx=+1, ny=-1, nz=+1
    )  # neighbor of the *next slide* with on the *top height and right position*
    n_25 = neighbour(
        coord, nx=+1, ny=0, nz=+1
    )  # neighbor of the *next slide* with on the *same height and tight position*
    n_26 = neighbour(
        coord, nx=+1, ny=+1, nz=+1
    )  # neighbor of the *next slide* with on the *bottom height and tight position*

    if connect == 4:  # 2D, only surfaces connected
        # connect_voxels_coordinates = np.concatenate((n_11,n_13,n_14,n_16), axis=0)
        connect_voxels_coordinates = np.vstack((n_11, n_13, n_14, n_16))

    elif connect == 8:  # 2D, surfaces & corners connected
        # connect_voxels_coordinates = np.concatenate((n_10,n_11,n_12,n_13,n_14,n_15,n_16,n_17), axis=0)
        connect_voxels_coordinates = np.vstack((n_10, n_11, n_12, n_13, n_14, n_15, n_16, n_17))

    elif connect == 6:  # 3D, only surfaces
        # connect_voxels_coordinates = np.concatenate((n_5,n_11,n_13,n_14,n_16,n_22), axis=0)
        connect_voxels_coordinates = np.vstack((n_5, n_11, n_13, n_14, n_16, n_22))

    elif connect == 18:  # 3D, surfaces & edges, but no connection only by corners
        # connect_voxels_coordinates = np.concatenate((n_2, n_4, n_5, n_6,
        #                                        n_8, n_10, n_11, n_12, n_13, n_14,
        #                                        n_15, n_16, n_17, n_19, n_21,
        #                                        n_22, n_23, n_25), axis=0)
        connect_voxels_coordinates = np.vstack(
            (n_2, n_4, n_5, n_6, n_8, n_10, n_11, n_12, n_13, n_14, n_15, n_16, n_17, n_19, n_21, n_22, n_23, n_25)
        )

    elif connect == 26:  # 3D, surfaces & edges & corners
        # connect_voxels_coordinates = np.concatenate((n_1, n_2, n_3, n_4, n_5, n_6, n_7,
        #                                        n_8, n_9, n_10, n_11, n_12, n_13, n_14,
        #                                        n_15, n_16, n_17, n_18, n_19, n_20, n_21,
        #                                        n_22, n_23, n_24, n_25, n_26), axis=0)
        connect_voxels_coordinates = np.vstack(
            (
                n_1,
                n_2,
                n_3,
                n_4,
                n_5,
                n_6,
                n_7,
                n_8,
                n_9,
                n_10,
                n_11,
                n_12,
                n_13,
                n_14,
                n_15,
                n_16,
                n_17,
                n_18,
                n_19,
                n_20,
                n_21,
                n_22,
                n_23,
                n_24,
                n_25,
                n_26,
            )
        )

    else:
        ValueError("Connection type is not clear.")

    return connect_voxels_coordinates


def main_tort(
    _image: np.ndarray,
    _direction_flow: int,
    _starting_depth: int = 0,
    _connection_type: int = 26,
    _plotting_type: bool = False,
    _fname: str = "TestFig",
    _DIRname: str = "Test_Microstructures/",
):
    """
    Main Function to execute in order to find statistical tortuosity values


    _image :            voxel-based image (array)
    _direction_flow :   0 (x), 1 (y), 2 (z) (integer)
    _starting_depth :   depth of running paths
    _connection_type :  type of connection (6, 18, 26 [default])
    _plotting_type:     Plots of all steps? (default: False)
    _fname:             Filename
    _DIRname:           Name of directory



    tortuosity_mean :   mean tortuosity
    tortuosity_std :    standard deviation of tortuosity
    tortuosity_median : median of tortuosity
    """
    image_orig = np.copy(_image)

    # skeletonize image
    (
        medial_axis_image,
        nrows,
        ncolumns,
        ndepth,
        cum_number_medial_each_slide,
        number_medial_each_slide,
        skeleton_volume_axis_index,
    ) = skeletonize3D(_image, _direction_flow, _plot_skel=_plotting_type, _fname=_fname, _DIRname=_DIRname)

    # find junctions (prestep for finding starting points)
    junction_index = find_junctions(image_orig, ncolumns, nrows, ndepth)

    # find starting points
    SP_loc_rad_shuffle = find_starting_points(
        _image,
        junction_index,
        _starting_depth,
        nrows,
        ncolumns,
        _plot_SP=_plotting_type,
        _fname=_fname,
        _DIRname=_DIRname,
    )

    # find lengths of paths
    all_paths_length = running_paths(
        _image,
        SP_loc_rad_shuffle,
        nrows,
        ncolumns,
        ndepth,
        cum_number_medial_each_slide,
        number_medial_each_slide,
        skeleton_volume_axis_index,
        _connect_search=_connection_type,
        _plot=False,
        _fname=_fname,
        _DIRname=_DIRname,
    )

    # get statistical values
    tortuosity_mean, tortuosity_std, tortuosity_median = statistical_tort(
        all_paths_length, nrows, _plot=_plotting_type, _SP_loc_rad=SP_loc_rad_shuffle, _fname=_fname, _DIRname=_DIRname
    )

    # no use of plotting animations - not necessary here

    return tortuosity_mean, tortuosity_std, tortuosity_median  # , all_paths_length
