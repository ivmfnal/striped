#include "Python.h"
#include <math.h>
#include "numpy/ndarraytypes.h"
#include "numpy/npy_3kcompat.h"

static PyObject *
py_pairs(PyObject *dummy, PyObject *args)
{
    /*
    *   Arguments:
    *       stripe - ndarray, shape = (n,)
    *       size_array - ndarray, shape = (e,)
    *   Returns pairs array shape = (2, ...) and pairs_size array (n,)
    */

    PyObject *sp = NULL, *zp = NULL;
    
    if (!PyArg_ParseTuple(args, "OO", &sp, &zp)) return NULL;

    PyArrayObject *stripe = (PyArrayObject*)sp;
    PyArrayObject *size = (PyArrayObject*)zp;
    PyArrayObject *pairs = NULL;
    PyArrayObject *pair_sizes = NULL;

    /*npy_intp *stripe_shape = PyArray_SHAPE(stripe);*/
    npy_intp *size_shape = PyArray_SHAPE(size);
    int type_num = PyArray_DESCR(stripe)->type_num;
    int item_size = PyArray_ITEMSIZE(stripe);
    int nevents = size_shape[0];
    
    int stride_size = PyArray_STRIDE(size, 0);
    int stride_stripe = PyArray_STRIDE(stripe, 0);

    /*
     * Calculate output array shape
     */
     
    int m = 0;
    int ievent;
    void *size_p = PyArray_GETPTR1(size, 0);
    for( ievent=0; ievent<nevents; ievent++ )
    {
        int n = *(int*)size_p;
        if ( n >= 2 )     
             m += n*(n-1)/2;
        size_p += stride_size;
    }
    
    npy_intp outshape[2];
    outshape[0] = 2;
    outshape[1] = m;
    
    npy_intp pair_size_shape[1];
    pair_size_shape[0] = nevents;

    pairs = (PyArrayObject *) PyArray_SimpleNew(2, outshape, type_num);
    if( pairs == NULL ) goto fail;

    pair_sizes = (PyArrayObject *) PyArray_SimpleNew(1, pair_size_shape, NPY_LONG);
    if( pair_sizes == NULL ) goto fail;

    int stride_out_0 = PyArray_STRIDE(pairs, 0);
    int stride_out_1 = PyArray_STRIDE(pairs, 1);
    
    size_p = PyArray_GETPTR1(size, 0);
    void *segment_p = PyArray_GETPTR1(stripe, 0);
    void *pairs_p = PyArray_GETPTR2(pairs, 0, 0);
    void *p0 = pairs_p;
    void *p1 = pairs_p + stride_out_0;
    void *pair_size_p = PyArray_GETPTR1(pair_sizes, 0);
    int pair_size_stride = PyArray_STRIDE(pair_sizes, 0);
    
    for( ievent=0; ievent<nevents; ievent++ )
    {
        int nitems = *(int*)size_p;
        int npairs = nitems*(nitems-1)/2;
        int j;
        
        for( j = 0; j < nitems-1; j++ )
        {
            void *stripe_p = segment_p + j*stride_stripe;
            int ntail = nitems - j - 1;
            int kk;
            for( kk = 0; kk < ntail; kk++ )
            {
                memcpy(p0, stripe_p, item_size);
                p0 += stride_out_1;
            }
            stripe_p += stride_stripe;
            for( kk = 0; kk < ntail; kk++ )
            {
                memcpy(p1, stripe_p, item_size);
                p1 += stride_out_1;
                stripe_p += stride_stripe;
            }
        }

        segment_p += nitems * stride_stripe;

        *(long*)pair_size_p = npairs;
        pair_size_p += pair_size_stride;
        
        size_p += stride_size;
    }
    
    /*
    Py_DECREF(stripe);
    Py_DECREF(size);
    */
    PyObject *out_tuple = Py_BuildValue("(OO)", (PyObject*)pairs, (PyObject*)pair_sizes);
    Py_DECREF(pairs);
    Py_DECREF(pair_sizes);
    return out_tuple;
    
fail:
    /*
    Py_XDECREF(stripe);
    Py_XDECREF(size);
    */
    Py_XDECREF(pairs);
    Py_XDECREF(pair_sizes);
    return NULL;
}

static PyMethodDef module_methods[] = {
    {"pairs", (PyCFunction) py_pairs, METH_VARARGS, "Make pairs from a stripe"},
    {NULL}  /* Sentinel */
};

    
    
#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif

PyMODINIT_FUNC
initstriped_c_tools(void) 
{
    PyObject* m;

    m = Py_InitModule3("striped_c_tools", module_methods,
                       "Low level stripe handling library");
    import_array();
}
