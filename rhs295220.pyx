# This file is generated automatically by QuTiP.
# (C) 2011 and later, QuSTaR

import numpy as np
cimport numpy as np
cimport cython
np.import_array()
cdef extern from "numpy/arrayobject.h" nogil:
    void PyDataMem_NEW_ZEROED(size_t size, size_t elsize)
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

from qutip.cy.spmatfuncs cimport spmvpy
from qutip.cy.interpolate cimport interp, zinterp
from qutip.cy.math cimport erf
cdef double pi = 3.14159265358979323

include '/opt/conda/lib/python3.6/site-packages/qutip/cy/complex_math.pxi'



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def cy_td_ode_rhs(
        double t,
        complex[::1] vec,
        complex[::1] data0,int[::1] idx0,int[::1] ptr0,
        int t0,
        float t1,
        float t2,
        float t3,
        float t4,
        float t5,
        float t6,
        float tg,
        int Q,
        float Omega):
    
    cdef size_t row
    cdef unsigned int num_rows = vec.shape[0]
    cdef double complex * out = <complex *>PyDataMem_NEW_ZEROED(num_rows,sizeof(complex))
     
    spmvpy(&data0[0], &idx0[0], &ptr0[0], &vec[0], 1.0, out, num_rows)
    cdef np.npy_intp dims = num_rows
    cdef np.ndarray[complex, ndim=1, mode='c'] arr_out = np.PyArray_SimpleNewFromData(1, &dims, np.NPY_COMPLEX128, out)
    PyArray_ENABLEFLAGS(arr_out, np.NPY_OWNDATA)
    return arr_out   

