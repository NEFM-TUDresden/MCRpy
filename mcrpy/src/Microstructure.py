from contextlib import contextmanager
import itertools
import logging
import pickle
import random
from typing import Dict, List, Tuple, Union

import numpy as np
from scipy.ndimage import convolve
import tensorflow as tf
from contextlib import suppress
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from mcrpy.src.IndicatorFunction import IndicatorFunction
from mcrpy.src.Symmetry import Symmetry, Cubic

LOADED_GRIDTOVTK = False
with suppress(Exception):
    from pyevtk.hl import gridToVTK
    LOADED_GRIDTOVTK = True

class Microstructure:

    def __init__(self, 
            array: np.ndarray, 
            use_multiphase = False,
            ori_repr: type = None,
            symmetry: Symmetry = Cubic,
            skip_encoding: bool = False,
            trainable: bool = True):
        """
        Creates a Microstructure object from a numpy array. The Microstructure
        object internally hold a tf.Variable called x, which contains all information
        and with regard to which all gradients are computed. This array has the
        following shape, depending on the situation:
        1, I, J, (K,) n_phases # if has_phases and use_multiphase
        1, I, J, (K,) 1        # if has_phases and not use_multiphase
        1, I, J, (K,) 3        # if has_orientations 
        Where I, J, and K are the number of pixels or voxels in x, y and z direction
        (if 3D), respectively. The first dimension is the batch dimension as needed
        for TensorFlow. Normally, self.x is encoded accordingly to use_multiphase from
        the input array with shape I, J(, K) if has_phases or I, J(, K), 3 if
        has_orientations. This requires phase_arrays to be given in a form where each
        pixel or voxel contains the integer value of the corresponding phase number.
        If a multiphase encoding with multiple, possibly real-valued indicator functions
        is preferred, you can set skip_encoding to True. If you do this, you better
        know what you are doing! Finally, if you will never need to compute gradients
        with respect to the microstructure, you can set trainable to False.
        """
        if -0.0001 < np.min(array) < 0:
            array[array < 0] = 0
        if 1.0 < np.max(array) < 1.0001:
            array[array > 1] = 1
        if np.sum(np.isnan(array)) > 0:
            logging.warning('Array to initialize microstructure contains NANs! - filling with zeros')
            array[np.isnan(array)] = 0
        if skip_encoding and (use_multiphase or array.shape[-1] == 1):
            logging.info('Skipping encoding')
            logging.info('Skip encoding, assume phase information')
            self.n_phases = array.shape[-1]
            self.phase_numbers = list(range(self.n_phases))
            if array.shape[0] == 1:
                array = array[0]
            x_np = array.astype(np.float64).clip(0, 1)
            assert len(x_np.shape) in {3, 4}
            self.is_3D = len(x_np.shape) == 4
            self.has_orientations = False
        elif array.shape[-1] == 3 and not use_multiphase:
            assert len(array.shape) in {3, 4}
            logging.info('Assume orientation information')
            self.phase_numbers = [0]
            self.n_phases = 1
            x_np = symmetry.project_to_fz(ori_repr(array.astype(np.float64))).x.numpy()
            self.is_3D = len(array.shape) == 4
            self.has_orientations = True
        elif skip_encoding:
            raise ValueError("Cannot skip encoding.")
        elif use_multiphase or np.max(array) > 1:
            logging.info('Assume phase information and use multiphase')
            phases_int = np.round(array).astype(np.int8)
            assert np.max(np.abs(phases_int-array)) < 1e-11
            phases = phases_int

            self.phase_numbers = np.unique(phases)
            logging.info(f'encoding phases {self.phase_numbers}')
            self.n_phases = len(self.phase_numbers)
            if not all(np.array(list(range(self.n_phases))) == self.phase_numbers):
                raise ValueError('Phases should be numbered consecutively, starting at 0.')
            encoded_ms = np.zeros((*phases.shape, self.n_phases), np.int8)
            for phase_number in self.phase_numbers:
                encoded_ms[..., phase_number] = phases == phase_number
            x_np = encoded_ms
            assert len(phases.shape) in {2, 3}
            self.is_3D = len(phases.shape) == 3
            self.has_orientations = False
        else:
            logging.info('Assume phase information and use singlephase')
            assert 0 <= np.min(array)
            assert 1 >= np.max(array)
            self.n_phases = 1
            self.phase_numbers = [0]
            x_np = array.reshape((*array.shape, 1)).astype(np.float64)
            assert len(array.shape) in {2, 3}
            self.is_3D = len(array.shape) == 3
            self.has_orientations = False
        self.has_phases = not self.has_orientations
        self.is_2D = not self.is_3D
        self.symmetry = symmetry

        self.shape = x_np.shape
        self.spatial_shape = self.shape[:-1]
        self.extra_shape = self.shape[-1]
        self.x_shape = tuple([1] + list(self.shape))

        if self.is_3D:
            self.paddings = tf.constant(np.zeros((4, 2), dtype=np.int32), dtype=tf.int32)
            block_shapes_np = np.zeros((3, 4), dtype=np.int32)
            for n_dim, dim_shape in enumerate(self.spatial_shape):
                block_shape = np.ones(4)
                block_shape[n_dim] = dim_shape
                block_shapes_np[n_dim] = block_shape
            self.block_shapes = tf.constant(block_shapes_np, dtype=tf.int32)
            self.batch_element_shapes = [
                    (1, self.spatial_shape[1], self.spatial_shape[2], self.extra_shape),
                    (1, self.spatial_shape[0], self.spatial_shape[2], self.extra_shape),
                    (1, self.spatial_shape[0], self.spatial_shape[1], self.extra_shape),
                    ]
            self.swapped_index_1 = tf.Variable([0, 0, 0], trainable=False, dtype=tf.int32)
            self.swapped_index_2 = tf.Variable([0, 0, 0], trainable=False, dtype=tf.int32)

        self.ori_repr = ori_repr
        self.x = tf.Variable(initial_value=x_np.reshape(self.x_shape).astype(np.float64), trainable=trainable, dtype=tf.float64, name='microstructure')

    @property
    def xx(self):
        return self.ori if self.has_orientations else self.indicator_function.x

    @property
    def indicator_function(self):
        assert self.has_phases
        return IndicatorFunction(self.x)

    @property
    def ori(self):
        assert self.has_orientations
        return self.ori_repr(self.x)

    @classmethod
    def load(cls, filename: str , use_multiphase: bool = False, trainable: bool = True):
        """Load Microstructure from npy-file (by calling the constructor) or from pickle-file,
        in which case the pickled Microstructure object is returned and __init__ is not called.
        Note that the kwargs use_multiphase and trainable are only used if the Microstructure
        is loaded from a npy-file. Internally, this function merely checks the filename ending
        and calls Microstructure.from_npy or Microstructure.from_pickle.
        """
        if filename.endswith('.npy'):
            ms = cls.from_npy(filename, use_multiphase=use_multiphase, trainable=trainable)
        elif filename.endswith('.pickle'):
            logging.info('Loading from pickle, hence ignoring further kwargs.')
            ms = cls.from_pickle(filename)
        else:
            raise NotImplementedError('Filetype not supported')
        return ms

    @classmethod
    def from_pickle(cls, filename: str):
        """Load a Microstructure from a pickle file, assert that it is indeed a Microstructure
        and return it.
        """
        with open(filename, 'rb') as f:
            ms = pickle.load(f)
        assert isinstance(ms, cls)
        return ms

    @classmethod
    def from_npy(cls, filename: str, use_multiphase: bool = False, trainable: bool = True, ori_repr: type = None):
        """Load a Microstructure from a numpy-array stored in a npy-file by loading the array
        and calling the constructor on it. The arguments use_multiphase and trainable are
        passed to the Microstructure constructor, so please refer to the class documentation
        for their meaning.
        """
        array = np.load(filename)
        return cls(array, use_multiphase=use_multiphase, trainable=trainable, ori_repr=ori_repr)

    def save(self, filename: str):
        logging.info(f'saving microstructure to {filename}')
        if filename.endswith('.npy'):
            self.to_npy(filename)
        elif filename.endswith('.pickle'):
            self.to_pickle(filename)
        else:
            raise NotImplementedError('Filetype not supported')

    def to_damask(self, filename: str):
        assert filename.endswith('.vti')
        import damask
        ms = self.decode_phases().reshape(self.spatial_shape)
        grid = (1, 1, 1) if self.is_3D else (1, 1)
        damask_grid = damask.Grid(ms, grid)
        damask_grid.save(filename[:-4])

    def to_pickle(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def to_npy(self, filename: str):
        """Save to numpy file"""
        if self.has_phases:
            microstructure = self.decode_phases()
        else:
            microstructure = self.get_orientation_field().numpy()

        np.save(filename, microstructure)

    def to_paraview(self, filename: str):
        assert LOADED_GRIDTOVTK, 'Cannot export to paraview because gridToVTK import failed - please install optional dependency gridToVTK'
        logging.info(f'Exporting to {filename}')
        coords_x = np.arange(0, self.shape[0] + 1)
        coords_y = np.arange(0, self.shape[1] + 1)
        coords_z = np.arange(0, self.shape[2] + 1)
        if self.has_phases:
            cellData = {'phase_ids': self.decode_phases(raw=True)}
        else:
            raise NotImplementedError()

        gridToVTK(filename[:-4], coords_x, coords_y, coords_z, cellData = cellData)

    def __repr__(self):
        representation = f"""MCRpy Microstructure object at {id(self)} 
            with spatial resolution {self.spatial_shape} ({'3D' if self.is_3D else '2D'}),"""
        if self.has_phases:
            representation += f"""
            using {max(self.n_phases, 2)} phases in {'single' if self.n_phases == 1 else 'multi'}phase representation."""
        if self.has_orientations:
            representation += f"""
            using {self.ori_repr} for orientation."""
        return representation

    def __iter__(self):
        """Returns an iterator over indices of spatial fields. Usage:
        for spatial_index in microstructure:
            # do something
            raise NotImplementedError() """
        return itertools.product(*[range(e) for e in self.spatial_shape])


    @contextmanager
    def use_singlephase_encoding(self):
        """Context manager to directly manipulate x, where x is given in singlephase encoding. This can be useful
        if certain operations are easier to define in singlephase notation. The usage is as follows:
        with microstructure.use_singlephase_encoding() as x:
            # do something with x
            raise NotImplementedError()
        After de-indentation, the local variable x from within the context manager is transformed to the original
        representation (if needed) and assigned to microstructure.x . For an example usage, see Microstructure.mutate.
        """
        assert self.has_phases
        was_singlephase = self.n_phases == 1
        x = self.indicator_function.as_singlephase().x
        x = tf.Variable(x, trainable=False)
        yield x
        if not was_singlephase:
            x = IndicatorFunction(x).was_multiphase().x
        self.x.assign(x)

    @contextmanager
    def use_multiphase_encoding(self):
        """Context manager to directly manipulate x, where x is given in multiphase encoding. This can be useful
        if certain operations are easier to define in multiphase notation. The usage is as follows:
        with microstructure.use_multiphase_encoding() as x:
            # do something with x
            raise NotImplementedError()
        After de-indentation, the local variable x from within the context manager is transformed to the original
        representation (if needed) and assigned to microstructure.x . For an example usage, see Microstructure.mutate.
        """
        assert self.has_phases
        was_multiphase = self.n_phases > 1
        x = self.indicator_function.as_multiphase().x
        x = tf.Variable(x, trainable=False)
        yield x
        if not was_multiphase:
            x = IndicatorFunction(x).as_singlephase().x
        self.x.assign(x)

    def numpy(self):
        return self.x[0].numpy()

    def get_orientation_field(self):
        return self.ori_repr(self.x[0])

    def get_full_field(self, phase_number):
        assert self.has_phases
        if self.is_3D:
            return self.x[0, :, :, :, phase_number]
        else:
            return self.x[0, :, :, phase_number]

    def get_slice(self, dimension: int, slice_index: int):
        assert self.is_3D
        batch_element_shape = self.batch_element_shapes[dimension]
        x_s2b = tf.space_to_batch(self.x, self.block_shapes[dimension], self.paddings)
        x_e_reshaped = tf.reshape(x_s2b[slice_index], batch_element_shape)
        x_e_reshaped.set_shape(batch_element_shape)
        if self.has_orientations:
            x_e_reshaped = self.ori_repr(x_e_reshaped) # TODO hier drehen je nach x y z mgl
        return x_e_reshaped

    def get_slice_iterator(self, dimension: int):
        assert self.is_3D 
        def my_generator(dimension):
            for slice_number in range(self.spatial_shape[dimension]):
                yield self.get_slice(dimension, slice_number)
        return my_generator(dimension)


    def decode_phase_array(self, phase_array: tf.Tensor, specific_phase: int = None, raw: bool = False) -> np.ndarray:
        assert self.has_phases
        if phase_array.shape[0] == 1:
            phase_array = phase_array[0]
        if self.n_phases == 1:
            result = phase_array.numpy()
            if result.shape[-1] == 1:
                result = result[..., 0]
            return result if raw else np.round(result)
        if specific_phase is not None:
            assert specific_phase in list(range(self.n_phases))
            result = phase_array.numpy()[..., specific_phase]
            return result if raw else np.round(result)
        array_np = phase_array.numpy()
        n_entries = np.product(array_np.shape) // self.n_phases
        array_reshaped = array_np.reshape((n_entries, -1))
        array_decoded = np.zeros(n_entries)
        for pixel in range(n_entries):
            array_decoded[pixel] = np.argmax(array_reshaped[pixel])
        return array_decoded.reshape(array_np.shape[:-1])

    def decode_phases(self, specific_phase: int = None, raw: bool = False) -> np.ndarray:
        assert self.has_phases
        return self.decode_phase_array(self.x, specific_phase=specific_phase, raw=raw)

    def decode_slice(self, dimension: int, slice_index: int, specific_phase: int = None, raw: bool = False):
        assert self.has_phases
        slice_to_decode = self.get_slice(dimension, slice_index)
        return self.decode_phase_array(slice_to_decode, specific_phase=specific_phase, raw=raw)
