"""
Interface for compute backends.
"""
import fmm.backend.numba as numba_backend

BACKEND = {
    "numba": {
        "p2m": numba_backend.p2m,
        "m2m": numba_backend.m2m,
        "m2l": numba_backend.m2l,
        "l2l": numba_backend.l2l,
        "s2l": numba_backend.s2l,
        "l2t": numba_backend.l2t,
        "m2t": numba_backend.m2t,
        "near_field": numba_backend.near_field,
    }
}
