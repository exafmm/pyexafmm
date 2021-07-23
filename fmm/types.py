"""
Custom Numpy and Numba compatible containers for FMM computations.
"""
from numba import njit, generated_jit
import numpy as np


class Arrayf4(np.ndarray):
    """
    Float32 Numpy Array
    """
    def __new__(
        subtype, shape, dtype=np.float32
    ):
        obj = super().__new__(subtype, shape, dtype)

        return obj

    def __array_finalize__(self, obj):
        if obj is None: return

from numba.types import Number, Float, Array
from functools import total_ordering

@total_ordering
class Float32(Number):
    def __init__(self, *args, **kws):
        super(Float32, self).__init__(*args, **kws)
        # Determine bitwidth
        assert self.name.startswith('float32')
        bitwidth = int(self.name[5:])
        self.bitwidth = bitwidth

    def cast_python_value(self, value):
        return getattr(np, self.name)(value)

    def __lt__(self, other):
        if self.__class__ is not other.__class__:
            return NotImplemented
        return self.bitwidth < other.bitwidth

f32_type = Float32('float32')

from numba.extending import typeof_impl

@typeof_impl.register(Float)
def typeof_index(val, c):
    return f32_type


@generated_jit(nopython=True)
def test(a):
    if isinstance(a, Float):
        if int(a.name[5:]) == 32:
            return lambda a: 'foo'
        else:
            return lambda a : 'bar'

    elif isinstance(a, Array):
        print(a.dtype)
        return lambda a: 'baz'
