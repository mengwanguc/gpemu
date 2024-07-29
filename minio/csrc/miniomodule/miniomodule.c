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

#include "../minio/minio.h"
#include <stdlib.h>
#include <pthread.h>

#define BLOCK_SIZE (4096)


/* Python wrapper for cache_t type. */
typedef struct {
    PyObject_HEAD
    cache_t *cache;                     /* MinIO cache. */
    size_t   max_usable_file_size;      /* Max file size we can read. */
    size_t   max_cacheable_file_size;   /* Max file size we can cache. Defaults
                                           to MAX_USABLE_FILE_SIZE if zero. */
    uint8_t *temp;                      /* MAX_USABLE_FILE_SIZE bytes used for
                                           copying. */
} PyCache;

/* PyCache deallocate method. */
static void
PyCache_dealloc(PyObject *self)
{
    PyCache *cache = (PyCache *) self;
    if (cache == NULL) {
        return;
    }

    /* Destroy the MinIO cache. */
    if (cache->cache != NULL) {
        cache_destroy(cache->cache);
        munmap(cache->cache, sizeof(cache_t));
    }

    /* Free the memory allocated for the copy region. */
    if (cache->temp != NULL) {
        free(cache->temp);
    }

    /* Free the cache wrapper struct itself. */
    Py_TYPE(cache)->tp_free((PyObject *) cache);
}

/* PyCache initialization method. */
static int
PyCache_init(PyObject *self, PyObject *args, PyObject *kwds)
{
    PyCache *cache = (PyCache *) self;

    /* Parse arguments. */
    size_t size, max_usable_file_size;
    size_t max_cacheable_file_size = 0; /* If zero, defaults to MAX_USABLE_FILE_SIZE. */
    size_t average_file_size = 0;
    static char *kwlist[] = {
        "size", "max_usable_file_size", "max_cacheable_file_size",
        "average_file_size", NULL
    };
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "kk|kk", kwlist,
                                     &size,
                                     &max_usable_file_size,
                                     &max_cacheable_file_size,
                                     &average_file_size)) {
        PyErr_SetString(PyExc_Exception, "missing/invalid argument");
        return -1;
    }

    /* Default to max usable file size (i.e., no-op). */
    if (max_cacheable_file_size == 0) {
        max_cacheable_file_size = max_usable_file_size;
    }

    /* If specified, the max usable item size must be <= max cacheable size. */
    if (max_cacheable_file_size > max_usable_file_size) {
        PyErr_SetString(PyExc_Exception, "max_cacheable_file_size must be <= max_usable_file_size");
        return -1;
    }

    /* Set up the cache struct as shared memory. */
    cache->cache = mmap_alloc(sizeof(cache_t));
    if (cache == NULL) {
        PyErr_SetString(PyExc_MemoryError, "unable to allocate cache_t struct");
        return -1;
    }

    /* Set up the copy area. */
    cache->max_usable_file_size = max_usable_file_size;
    cache->max_cacheable_file_size = max_cacheable_file_size;
    if (posix_memalign((void **) &cache->temp, BLOCK_SIZE, max_usable_file_size) != 0) {
        PyErr_SetString(PyExc_MemoryError, "couldn't allocate temp area");
        PyErr_NoMemory();
        return -1;
    }

    /* Initialize the cache. */
    int status = cache_init(cache->cache,
                            size,
                            max_cacheable_file_size,
                            average_file_size,
                            POLICY_MINIO);
    if (status < 0) {
        switch (status) {
            case -ENOMEM:
                PyErr_SetString(PyExc_MemoryError, "couldn't allocate cache");
                break;
            case -EPERM:
                PyErr_SetString(PyExc_PermissionError, "couldn't pin cache memory");
                break;
        }

        return -1;
    }

    return 0;
}

/* PyCache creation method. */
static PyObject *
PyCache_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    /* Allocate the PyCache struct. */
    PyCache *self;
    if ((self = (PyCache *) type->tp_alloc(type, 0)) == NULL) {
        PyErr_NoMemory();
        return NULL;
    }

    return (PyObject *) self;
}

/* PyCache method to check if a filepath is cached. */
static PyObject *
PyCache_contains(PyCache *self, PyObject *args, PyObject *kwds)
{
    /* Parse arguments. */
    char *filepath;
    static char *kwlist[] = {"filepath", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &filepath)) {
        PyErr_SetString(PyExc_Exception, "missing/invalid argument");
        return NULL;
    }

    return PyBool_FromLong((long) cache_contains(self->cache, filepath));
}

/* PyCache method to explicitly cache data. Returns True on success, False on
   failure. */
static PyObject *
PyCache_store(PyCache *self, PyObject *args, PyObject *kwds)
{
    /* Parse arguments. */
    char *filepath;
    size_t bytes;
    Py_buffer buf;
    static char *kwlist[] = {"filepath", "bytes", "data",  NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "sks*", kwlist, &filepath, &bytes,&buf)) {
        PyErr_SetString(PyExc_Exception, "missing/invalid argument");
        return NULL;
    }

    /* Don't cache things that are bigger than we allow. */
    if (bytes > self->max_cacheable_file_size) {
        return PyBool_FromLong(0);
    }


    int status = cache_store(self->cache, filepath, buf.buf, bytes);
    if (status < 0) {
        return PyBool_FromLong(0);
    }

    return PyBool_FromLong(1);
}

/* PyCache method to load from cache without issuing IO on miss. Returns a tuple
   (data, size) on success. Returns None on failure. */
static PyObject *
PyCache_load(PyCache *self, PyObject *args, PyObject *kwds)
{
    /* Parse arguments. */

    char *filepath;
    static char *kwlist[] = {"filepath", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &filepath)) {
        PyErr_SetString(PyExc_Exception, "missing/invalid argument");
        return NULL;
    }

    size_t size = 0;
    int status = cache_load(self->cache,
                            filepath,
                            self->temp,
                            &size,
                            self->max_cacheable_file_size);
    if (status < 0) {
        PyErr_Format(PyExc_Exception, "load failed; %s", strerror(-status));
        return NULL;
    }

    PyObject *bytes = PyBytes_FromStringAndSize((char *) self->temp, size);
    PyObject *size_ = PyLong_FromLong(size);
    PyObject *out = PyTuple_Pack(2, bytes, size_);

    /* Because PyTuple_Pack increments the reference counter for all inputs,
       we must decrement the refcounts to prevent a leak where the count is >1
       when we return. */
    Py_DECREF(bytes);
    Py_DECREF(size_);

    return out;
}

/* PyCache read/get method. Returns (data, size) as a tuple. */
static PyObject *
PyCache_read(PyCache *self, PyObject *args, PyObject *kwds)
{
    /* Parse arguments. */
    char *filepath;
    static char *kwlist[] = {"filepath", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &filepath)) {
        PyErr_SetString(PyExc_Exception, "missing/invalid argument");
        return NULL;
    }

    /* Get the file contents. */
    ssize_t size = cache_read(self->cache,
                              filepath,
                              self->temp,
                              self->max_usable_file_size);
    if (size < 0) {
        switch (size) {
            case -EINVAL:
                PyErr_SetString(PyExc_MemoryError, "insufficient buffer size");
                break;
            case -ENOMEM:
                PyErr_SetString(PyExc_MemoryError, "unable to allocate hash table entry");
                break;
            case -ENOENT:
                PyErr_SetString(PyExc_FileNotFoundError, filepath);
                break;
            default:
                PyErr_SetString(PyExc_Exception, "unknown exception");
                break;
        }

        return NULL;
    }

    PyObject *bytes = PyBytes_FromStringAndSize((char *) self->temp, size);
    PyObject *size_ = PyLong_FromLong(size);
    PyObject *out = PyTuple_Pack(2, bytes, size_);

    /* Because PyTuple_Pack increments the reference counter for all inputs,
       we must decrement the refcounts to prevent a leak where the count is >1
       when we return. */
    Py_DECREF(bytes);
    Py_DECREF(size_);

    return out;
}

/* PyCache method to flush the cache. */
static PyObject *
PyCache_flush(PyCache *self, PyObject *args, PyObject *kwds)
{
    cache_flush(self->cache);

    return PyLong_FromLong(0L);
}

/* PyCache method to get the cache's "size" field. */
static PyObject *
PyCache_get_size(PyCache *self, PyObject *args, PyObject *kwds)
{
    return PyLong_FromUnsignedLong(self->cache->size);
}

/* PyCache method to get the cache's "used" field. */
static PyObject *
PyCache_get_used(PyCache *self, PyObject *args, PyObject *kwds)
{
    size_t used = self->cache->used;

    return PyLong_FromUnsignedLong(used);
}

/* PyCache methods. */
static PyMethodDef PyCache_methods[] = {
    {
        "contains",
        (PyCFunction) PyCache_contains,
        METH_VARARGS | METH_KEYWORDS,
        "Check if the filepath is cached."
    },
    {
        "load",
        (PyCFunction) PyCache_load,
        METH_VARARGS | METH_KEYWORDS,
        "Load the filepath if it's cached (don't read on miss)."
    },
    {
        "store",
        (PyCFunction) PyCache_store,
        METH_VARARGS | METH_KEYWORDS,
        "Cache data at the given filepath."
    },
    {
        "read",
        (PyCFunction) PyCache_read,
        METH_VARARGS | METH_KEYWORDS,
        "Read a file through the cache."
    },
    {
        "flush",
        (PyCFunction) PyCache_flush,
        METH_NOARGS,
        "Flush the cache."
    },
    {
        "get_size",
        (PyCFunction) PyCache_get_size,
        METH_NOARGS,
        "Get size of cache in bytes."
    },
    {
        "get_used",
        (PyCFunction) PyCache_get_used,
        METH_NOARGS,
        "Get number of bytes used in cache."
    },
    {NULL} /* Sentinel. */
};

static PyTypeObject PythonCacheType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "minio.PythonCache",
    .tp_doc = PyDoc_STR("MinIO Python cache"),
    .tp_basicsize = sizeof(PyCache),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,

    /* Methods. */
    .tp_dealloc = PyCache_dealloc,
    .tp_new = PyCache_new,
    .tp_init = PyCache_init,
    .tp_methods = PyCache_methods,
};


static PyMethodDef CacheMethods[] = {
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef miniomodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "minio",
    .m_doc = "Python module to implement a MinIO file cache.",
    .m_size = -1,
    .m_methods = CacheMethods,
};

PyMODINIT_FUNC
PyInit_minio(void)
{
    PyObject *module;
    
    /* Check PythonCacheType is ready. */
    if (PyType_Ready(&PythonCacheType) < 0) {
        return NULL;
    }

    /* Create Python module. */
    if ((module = PyModule_Create(&miniomodule)) == NULL) {
        return NULL;
    }

    /* Add the PythonCacheType type. */
    Py_INCREF(&PythonCacheType);
    if (PyModule_AddObject(module, "PyCache", (PyObject *) &PythonCacheType) < 0) {
        Py_DECREF(&PythonCacheType);
        Py_DECREF(module);
        return NULL;
    }

    return module;
}