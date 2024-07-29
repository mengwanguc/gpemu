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

#include "../async/async.h"
#include "../utils/alloc.h"

#include <stdatomic.h>

/* Input validation. */
#define ARG_CHECK(valid_condition, error_string, return_fail)                  \
   if (!(valid_condition)) {                                                   \
      PyErr_SetString(PyExc_Exception, error_string);                          \
      return return_fail;                                                      \
   }

/* --------- */
/*   TYPES   */
/* --------- */

/* Python wrapper for entry_t struct. */
typedef struct {
   PyObject_HEAD

   entry_t *entry;
} Entry;

/* Python wrapper for wstate_t struct. */
typedef struct {
   PyObject_HEAD

   wstate_t *worker;
} Worker;

/* Python wrapper for lstate_t struct. */
typedef struct {
   PyObject_HEAD

   lstate_t *loader;    /* Asynchronous loader state. */
} Loader;


/* ------------------------ */
/*   LOADER ENTRY METHODS   */
/* ------------------------ */

/* Entry deallocate method. */
static void
Entry_dealloc(PyObject *self)
{
   Py_TYPE(self)->tp_free(self);
}

/* Entry allocation method. */
static PyObject *
Entry_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
   PyObject *entry;
   if ((entry = type->tp_alloc(type, 0)) == NULL) {
      PyErr_NoMemory();
      return NULL;
   }

   return (PyObject *) entry;
}

/* Entry initialization method. */
static int
Entry_init(PyObject *self, PyObject *args, PyObject *kwds)
{
   /* No-op. */
   return 0;
}

/* Get the filename for the data in this entry. */
static PyObject *
Entry_get_filepath(Worker *self, PyObject *args, PyObject *kwds)
{
   Entry *entry = (Entry *) self;

   return PyBytes_FromString(entry->entry->path);
}

/* Get the data in this entry. */
static PyObject *
Entry_get_data(Worker *self, PyObject *args, PyObject *kwds)
{
   Entry *entry = (Entry *) self;

   return PyBytes_FromStringAndSize((char *) entry->entry->shm_wdata, entry->entry->size);
}

/* Release an entry. */
static PyObject *
Entry_release(Worker *self, PyObject *args, PyObject *kwds)
{
   Entry *entry = (Entry *) self;

   /* Release the wrapped entry, unless it's NULL. */
   if (entry->entry == NULL) {
      PyErr_SetString(PyExc_Exception, "cannot release entry; empty wrapper");
      return NULL;
   }
   async_release(entry->entry);

   return PyLong_FromLong(0);
}

/* Entry methods array. */
static PyMethodDef Entry_methods[] = {
   {
      "get_filepath",
      (PyCFunction) Entry_get_filepath,
      METH_NOARGS,
      "Get the filepath for this entry."
   },
   {
      "get_data",
      (PyCFunction) Entry_get_data,
      METH_NOARGS,
      "Get the data contained by this entry."
   },
   {
      "release",
      (PyCFunction) Entry_release,
      METH_NOARGS,
      "Release (and discard) this entry."
   },
   {NULL}
};

/* Entry type declaration. */
static PyTypeObject PythonEntryType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "AsyncLoader.LoaderEntry",
    .tp_doc = PyDoc_STR("Loader entry"),
    .tp_basicsize = sizeof(Entry),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,

    /* Methods. */
    .tp_dealloc = Entry_dealloc,
    .tp_new = Entry_new,
    .tp_init = Entry_init,
    .tp_methods = Entry_methods,
};


/* ------------------ */
/*   WORKER CONTEXT   */
/* ------------------ */

/* Worker deallocate method. */
static void
Worker_dealloc(PyObject *self)
{
   Py_TYPE(self)->tp_free(self);
}

/* Worker allocation method. */
static PyObject *
Worker_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
   PyObject *worker;
   if ((worker = type->tp_alloc(type, 0)) == NULL) {
      PyErr_NoMemory();
      return NULL;
   }

   return (PyObject *) worker;
}

/* Worker initialization mmethod. */
static int
Worker_init(PyObject *self, PyObject *args, PyObject *kwds)
{
   /* No-op. */
   return 0;
}

/* Worker method to request a file be loaded. On success, returns True. On
   failure, returns False. */
static PyObject *
Worker_request(Worker *self, PyObject *args, PyObject *kwds)
{
   char *filepath;
   static char *kwlist[] = {"filepath", NULL};
   if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &filepath)) {
      PyErr_SetString(PyExc_Exception, "missing/invalid argument");
      return NULL;
   }

   if (!async_try_request(self->worker, filepath)) {
      return PyBool_FromLong(false);
   }

   return PyBool_FromLong(true);
}

/* Worker method to try to get a file. If a file is waiting in the completion
   queue, that file is returned and popped from the queue. Otherwise, None is
   returned. */
static PyObject *
Worker_try_get(Worker *self, PyObject *args, PyObject *kwds)
{
   /* Get an entry from the completion queue. */
   entry_t *e = async_try_get(self->worker);
   if (e == NULL) {
      Py_INCREF(Py_None);
      return Py_None;
   }

   /* Allocate a wrapper. */
   Entry *entry = (Entry *) Entry_new(&PythonEntryType, NULL, NULL);
   if (entry == NULL) {
      return NULL;
   }

   /* Insert the entry_t into the wrapper. */
   entry->entry = e;

   return (PyObject *) entry;
}

/* Worker method to block until a file is loaded. Removes the file from the
   completion queue, and returns it. */
static PyObject *
Worker_wait_get(Worker *self, PyObject *args, PyObject *kwds)
{
   /* Spin until we get an entry. */
   entry_t *e;
   while ((e = async_try_get(self->worker)) == NULL) {}

   /* Allocate a wrapper. */
   Entry *entry = (Entry *) Entry_new(&PythonEntryType, NULL, NULL);
   if (entry == NULL) {
      return NULL;
   }
   
   /* Insert the entry_t into the wrapper. */
   entry->entry = e;

   return (PyObject *) entry;
}

/* Worker methods array. */
static PyMethodDef Worker_methods[] = {
   {
      "request",
      (PyCFunction) Worker_request,
      METH_VARARGS | METH_KEYWORDS,
      "Request that a file be loaded."
   },
   {
      "try_get",
      (PyCFunction) Worker_try_get,
      METH_NOARGS,
      "Try to get a file, if one has been loaded."
   },
   {
      "wait_get",
      (PyCFunction) Worker_wait_get,
      METH_NOARGS,
      "Block until a file has been loaded."
   },
   {NULL}
};

/* Worker type declaration. */
static PyTypeObject PythonWorkerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "AsyncLoader.Worker",
    .tp_doc = PyDoc_STR("Worker context"),
    .tp_basicsize = sizeof(Worker),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,

    /* Methods. */
    .tp_dealloc = Worker_dealloc,
    .tp_new = Worker_new,
    .tp_init = Worker_init,
    .tp_methods = Worker_methods,
};


/* ------------------ */
/*   LOADER PROCESS   */
/* ------------------ */

/* Loader deallocate method. */
static void
Loader_dealloc(PyObject *self)
{
   Loader *loader = (Loader *) self;
   if (loader == NULL) {
      return;
   }

   if (loader->loader != NULL) {
      /* All of the lstate_t's shared memory is allocated contiguously in a
         single mmap, so freeing it is simple. */
      if (loader->loader->states != NULL) {
         mmap_free(loader->loader->states, loader->loader->total_size);
      }

      /* Free the lstate_t struct itself. */
      mmap_free(loader->loader, sizeof(lstate_t));
   }

   /* Free the Loader wrapper. */
   Py_TYPE(loader)->tp_free((PyObject *) loader);
}

/* Loader allocation method. */
static PyObject *
Loader_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
   /* Allocate the Loader struct. */
   PyObject *self;
   if ((self = type->tp_alloc(type, 0)) == NULL) {
      PyErr_NoMemory();
      return NULL;
   }

   return (PyObject *) self;
}

/* Loader initialization method. */
static int
Loader_init(PyObject *self, PyObject *args, PyObject *kwds)
{
   Loader *loader = (Loader *) self;

   /* Parse arguments. */
   int direct = 0;
   size_t queue_depth, n_workers, dispatch_n, max_idle_iters;
   static char *kwlist[] = {
      "queue_depth", "n_workers", "dispatch_n", "max_idle_iters", "direct", NULL
   };
   if (!PyArg_ParseTupleAndKeywords(args, kwds, "kkkk|p", kwlist,
                                    &queue_depth,
                                    &n_workers,
                                    &dispatch_n,
                                    &max_idle_iters,
                                    &direct)) {
      PyErr_SetString(PyExc_Exception, "missing/invalid argument");
      return -1;
   }

   /* Sanity-check arguments. */
   ARG_CHECK(queue_depth > 0, "queue depth must be positive", -1);
   ARG_CHECK(n_workers > 0, "must have >=1 worker(s)", -1);

   /* Allocate lstate using shared memory. */
   if ((loader->loader = mmap_alloc(sizeof(lstate_t))) == NULL) {
      PyErr_SetString(PyExc_Exception, "failed to allocate loader struct");
      return -1;
   }

   /* Initialize the loader. */
   int status = async_init(loader->loader,
                           queue_depth,
                           n_workers,
                           dispatch_n,
                           max_idle_iters,
                           direct ? __O_DIRECT : 0);
   if (status < 0) {
      PyErr_Format(PyExc_Exception,
                   "failed to initialize loader; %s",
                   strerror(-status));
      return -1;
   }

   return 0;
}

/* Loader method to become a loader process. */
static PyObject *
Loader_become_loader(Loader *self, PyObject *args, PyObject *kwds)
{
   /* Start the loader. */
   async_start(self->loader);

   return PyLong_FromLong(0);
}

/* Loader method to spawn a loader process. */
static PyObject *
Loader_spawn_loader(Loader *self, PyObject *args, PyObject *kwds)
{
   /* Fork. Child starts the loader. */
   if (fork() == 0) {
      async_start(self->loader);

      /* Never reached. */
      assert(false);
   }

   return PyLong_FromLong(0);
}

/* Loader method to get the context for the worker with the given ID. */
static PyObject *
Loader_get_worker_context(Loader *self, PyObject *args, PyObject *kwds)
{
   /* Parse the arguments. */
   size_t id;
   static char *kwlist[] = {"id", NULL};
   if (!PyArg_ParseTupleAndKeywords(args, kwds, "k", kwlist, &id)) {
      PyErr_SetString(PyExc_Exception, "missing/invalid argument");
      return NULL;
   }

   /* Sanity-check the arguments. */
   ARG_CHECK(id < self->loader->n_states, "invalid worker id", NULL);

   /* Allocate a wrapper. */
   Worker *worker;
   if ((worker = (Worker *) Worker_new(&PythonWorkerType, NULL, NULL)) == NULL) {
      return NULL;
   }

   /* Fill the wrapper. */
   worker->worker = &self->loader->states[id];

   return (PyObject *) worker;
}

/* Loader methods array. */
static PyMethodDef Loader_methods[] = {
   {
      "become_loader",
      (PyCFunction) Loader_become_loader,
      METH_NOARGS,
      "Become the loader process."
   },
   {
      "spawn_loader",
      (PyCFunction) Loader_spawn_loader,
      METH_NOARGS,
      "Fork and spawn a loader process as a child."
   },
   {
      "get_worker_context",
      (PyCFunction) Loader_get_worker_context,
      METH_VARARGS | METH_KEYWORDS,
      "Get context for specified worker."
   },
   {NULL}
};

/* Loader type declaration. */
static PyTypeObject PythonLoaderType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "AsyncLoader.Loader",
    .tp_doc = PyDoc_STR("Loader context"),
    .tp_basicsize = sizeof(Loader),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,

    /* Methods. */
    .tp_dealloc = Loader_dealloc,
    .tp_new = Loader_new,
    .tp_init = Loader_init,
    .tp_methods = Loader_methods,
};


/* --------------- */
/*   MODULE INIT   */
/* --------------- */

static PyMethodDef ModuleMethods[] = {
   {NULL},
};

static struct PyModuleDef AsyncLoadermodule = {
   PyModuleDef_HEAD_INIT,
   .m_name = "AsyncLoader",
   .m_doc = "Python module for asynchronous file loading.",
   .m_size = -1,
   .m_methods = ModuleMethods,
};

/* Register a Python type with a module. */
#define REGISTER_TYPE(module, name, type_addr)                                 \
   Py_INCREF(type_addr);                                                       \
   if (PyModule_AddObject(module, name, (PyObject *) type_addr) < 0) {         \
      Py_DECREF(type_addr);                                                    \
      Py_DECREF(module);                                                       \
      return NULL;                                                             \
   }

PyMODINIT_FUNC
PyInit_AsyncLoader(void)
{
   /* Create module. */
   PyObject *module;
   if ((module = PyModule_Create(&AsyncLoadermodule)) == NULL) {
      return NULL;
   }

   /* Ready all types. */
   if (PyType_Ready(&PythonEntryType)  < 0 ||
       PyType_Ready(&PythonWorkerType) < 0 ||
       PyType_Ready(&PythonLoaderType) < 0) {
      return NULL;
   }

   /* Register all types. */
   REGISTER_TYPE(module, "Entry", &PythonEntryType);
   REGISTER_TYPE(module, "Worker", &PythonWorkerType);
   REGISTER_TYPE(module, "Loader", &PythonLoaderType);

   return module;
}