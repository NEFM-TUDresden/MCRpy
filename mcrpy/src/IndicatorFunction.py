from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Tuple, Union

import numpy as np
import tensorflow as tf

tf.keras.backend.set_floatx("float64")


class IndicatorFunction(tf.experimental.ExtensionType):
    x: tf.Tensor # 1 I J (K) n_phases

    def __call__(self):
        return self.x

    def __getitem__(self, item):
        return self.x[item]

    @property
    def n_phases(self):
        return tf.shape(self.x)[-1]

    def as_multiphase(self) -> IndicatorFunction:
        """Return a multiphase encoding representation copy of x, which is not necessarily the same as x.
        If x is already in multiphase representation, x is returned, else a new TensorFlow variable is returned.
        Do not use this unless you know what you are doing. Instead, it is recommended to use the context managers
        Microstructure.use_multiphase_encoding and Microstructure.use_singlephase_encoding."""
        if self.n_phases > 1:
            return self
        x_1 = self.x
        x_0 = 1 - x_1
        x_con = tf.concat([x_0, x_1], axis=-1)
        return IndicatorFunction(x_con)

    def as_singlephase(self) -> IndicatorFunction:
        """Return a singlephase encoding representation copy of x, which is not necessarily the same as x.
        If x is already in singlephase representation, x is returned, else a new TensorFlow variable is returned.
        Do not use this unless you know what you are doing. Instead, it is recommended to use the context managers
        Microstructure.use_multiphase_encoding and Microstructure.use_singlephase_encoding."""
        if self.n_phases == 1:
            return self
        if self.n_phases > 2:
            raise ValueError('too many phases to drop indicator function representatiton')
        first_phase = tf.gather(self.x, [1], axis=-1)
        return IndicatorFunction(first_phase)

    def decode_phase_array(self, phase_array: tf.Tensor, specific_phase: int = None, raw: bool = False) -> np.ndarray:
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
        return self.decode_phase_array(self.x, specific_phase=specific_phase, raw=raw)

    def decode_slice(self, dimension: int, slice_index: int, specific_phase: int = None, raw: bool = False):
        slice_to_decode = self.get_slice(dimension, slice_index)
        return self.decode_phase_array(slice_to_decode, specific_phase=specific_phase, raw=raw)


if __name__ == "__main__":
    from mcrpy import Microstructure
    ms = Microstructure.from_npy('../../microstructures/pymks_ms_64x64.npy')
    ifs = IndicatorFunction(ms.x).as_singlephase()
    assert ifs.as_multiphase().as_singlephase() == ifs
