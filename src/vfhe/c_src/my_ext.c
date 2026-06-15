#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "my_ext.h"

static PyObject* my_ext_hello(PyObject* self, PyObject* args) {
    return PyUnicode_FromString("Hello from the C extension!");
}

static PyMethodDef MyExtMethods[] = {
    {"hello", my_ext_hello, METH_NOARGS, "Return a greeting from C."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef myextmodule = {
    PyModuleDef_HEAD_INIT,
    "vfhe._my_ext", 
    "A C extension module for vfhe.", 
    -1, 
    MyExtMethods
};

PyMODINIT_FUNC PyInit__my_ext(void) {
    return PyModule_Create(&myextmodule);
}