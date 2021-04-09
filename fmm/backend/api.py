"""
Interface for compute backends.
"""
import fmm.backend.numba as numba_backend
import fmm.backend.openmp as openmp_backend

BACKEND = {
    "numba": {
        "p2m": numba_backend.p2m,
        "m2m": numba_backend.m2m,
        "m2l": numba_backend.m2l,
        "l2l": numba_backend.l2l,
        "s2l": numba_backend.s2l,
        "l2t": numba_backend.l2t,
        "m2t": numba_backend.m2t,
        "near_field_u_list": numba_backend.near_field_u_list,
        "near_field_node": numba_backend.near_field_node,
    },
    "openmp": {
        "p2m": openmp_backend.p2m,
        "m2m": openmp_backend.m2m,
        "m2l": openmp_backend.m2l,
        "l2l": openmp_backend.l2l,
        "s2l": openmp_backend.s2l,
        "l2t": openmp_backend.l2t,
        "m2t": openmp_backend.m2t,
        "near_field_u_list": openmp_backend.near_field,
        "near_field_node": openmp_backend.near_field,
    }
}
