/* MIT License

   Copyright (c) 2023 Gus Waldspurger

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
   */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdlib.h>
#include <stdbool.h>
#include <sys/mman.h>

typedef struct {
    PyObject_HEAD

    long     used;      /* Whether this balloon is currently claimed. Completely
                           optional to use, and has no effect on operation of
                           the balloon. */
    size_t   size;      /* Size of BALLOON in bytes. */
    uint8_t *pinned;    /* Pinned memory. */
} PyBalloon;


/* PyBalloon deallocate method. */
static void
PyBalloon_dealloc(PyObject *self)
{
    PyBalloon *balloon = (PyBalloon *) self;

    /* Free the pinned memory. */
    if (balloon->pinned != NULL) {
        free(balloon->pinned);
    }

    /* Free the balloon struct itself. */
    Py_TYPE(balloon)->tp_free((PyObject *) balloon);
}

/* PyBalloon creation method. */
static PyObject *
PyBalloon_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    /* Allocate the PyBalloon struct. */
    PyBalloon *self;
    if ((self = (PyBalloon *) type->tp_alloc(type, 0)) == NULL) {
        PyErr_NoMemory();
        return NULL;
    }

    return (PyObject *) self;
}

/* PyBalloon initialization method. Creates a PyBalloon with SIZE bytes of
   pinned memory. */
static int
PyBalloon_init(PyObject *self, PyObject *args, PyObject *kwds)
{
    PyBalloon *balloon = (PyBalloon *) self;

    /* Parse arguments. */
    size_t size;
    static char *kwlist[] = {"size", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "k", kwlist, &size)) {
        PyErr_SetString(PyExc_Exception, "missing/invalid argument");
        return -1;
    }

    balloon->used = 0L;
    balloon->size = size;
    balloon->pinned = malloc(size);
    if (mlock(balloon->pinned, size) != 0) {
        PyErr_SetString(PyExc_PermissionError, "unable to pin balloon memory");
        return -1;
    }

    return 0;
}

static PyObject *
PyBalloon_get_size(PyBalloon *self, PyObject *args, PyObject *kwds)
{
    return PyLong_FromUnsignedLong(self->size);
}

static PyObject *
PyBalloon_get_used(PyBalloon *self, PyObject *args, PyObject *kwds)
{
    return PyBool_FromLong(self->used);
}

static PyObject *
PyBalloon_set_used(PyBalloon *self, PyObject *args, PyObject *kwds)
{
    /* Parse arguments. */
    long used;
    static char *kwlist[] = {"used", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "l", kwlist, &used)) {
        PyErr_SetString(PyExc_Exception, "missing/invalid argument");
        return NULL;
    }

    /* Update used with new value. */
    self->used = used;

    return PyLong_FromLong(0L);
}

/* PyBalloon methods. */
static PyMethodDef PyBalloon_methods[] = {
    {"get_size", (PyCFunction) PyBalloon_get_size, METH_NOARGS, "Get size of balloon's pinned region."},
    {"get_used", (PyCFunction) PyBalloon_get_used, METH_NOARGS, "Get balloon's used status."},
    {"set_used", (PyCFunction) PyBalloon_set_used, METH_VARARGS | METH_KEYWORDS, "Set balloon's used status."},
    {NULL}
};

/* Python Balloon type declaration. */
static PyTypeObject PythonBalloonType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "mlock.PythonBalloon",
    .tp_doc = PyDoc_STR("Pinned memory balloon"),
    .tp_basicsize = sizeof(PyBalloon),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,

    /* Methods. */
    .tp_dealloc = PyBalloon_dealloc,
    .tp_new = PyBalloon_new,
    .tp_init = PyBalloon_init,
    .tp_methods = PyBalloon_methods,
};

/* Module methods. */
static PyMethodDef module_methods[] = {
    {NULL, NULL, 0, NULL}
};

/* Module declaration. */
static struct PyModuleDef mlockmodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "mlock",
    .m_doc = "Python module to utilize page-locked memory.",
    .m_size = -1,
    .m_methods = module_methods,
};

/* Module initializer. */
PyMODINIT_FUNC
PyInit_mlock(void)
{
    PyObject *module;

    /* Check PythonCacheType is ready. */
    if (PyType_Ready(&PythonBalloonType) < 0) {
        return NULL;
    }

    /* Create Python module. */
    if ((module = PyModule_Create(&mlockmodule)) == NULL) {
        return NULL;
    }

    /* Add the PythonBalloonType type. */
    Py_INCREF(&PythonBalloonType);
    if (PyModule_AddObject(module, "PyBalloon", (PyObject *) &PythonBalloonType) < 0) {
        Py_DECREF(&PythonBalloonType);
        Py_DECREF(module);
        return NULL;
    }

    return module;
}