"""
Numba type aliases
"""
import numba

Single = numba.float32
Double = numba.float64
Int = numba.int32
Long = numba.int64
LongArray = numba.int64[:]
LongArray2D = numba.int64[:,:]
IntArray = numba.int32[:]
IntList = numba.types.ListType(Int)
LongIntList = numba.types.ListType(Long)
Coord = numba.float32[:]
Coords = numba.float32[:,:]
