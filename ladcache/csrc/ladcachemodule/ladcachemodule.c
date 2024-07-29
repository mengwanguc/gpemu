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

#include "../ladcache/cache.h"
#include "../utils/alloc.h"
#include "../utils/log.h"

/* Input validation. */
#define ARG_CHECK(valid_condition, error_string, return_fail)                  \
    do {                                                                       \
        if (!(valid_condition)) {                                              \
            PyErr_SetString(PyExc_Exception, error_string);                    \
            return return_fail;                                                \
        }                                                                      \
    } while (0)


/* --------- */
/*   TYPES   */
/* --------- */

/* Python ustate_t wrapper. */
typedef struct {
    PyObject_HEAD

    ustate_t *ustate;
} UserState;
static PyTypeObject PythonUserStateType;

/* Python cache_t wrapper. */
typedef struct {
    PyObject_HEAD

    cache_t *cache;
} Cache;
static PyTypeObject PythonCacheType;

/* Python request_t wrapper. */
typedef struct {
    PyObject_HEAD

    ustate_t  *ustate;
    request_t *request;
} Request;
static PyTypeObject PythonRequestType;


/* Generic object allocator. */
static PyObject *
Generic_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyObject *self;
    if ((self = type->tp_alloc(type, 0)) == NULL) {
        PyErr_NoMemory();
        return NULL;
    }

   return (PyObject *) self;
}

/* Generic object deallocator. */
static void
Generic_dealloc(PyObject *self)
{
    Py_TYPE(self)->tp_free(self);
}

/* Generic object no-op init. */
static int
Generic_init_noop(PyObject *self, PyObject *args, PyObject *kwds)
{
    /* No-op. */
    return 0;
}


/* --------------------  */
/*    `Request` METHODS    */
/* --------------------- */

/* Get the filepath this request loaded. */
PyObject *
Request_get_filepath(PyObject *self, PyObject *args, PyObject *kwds)
{
    return PyBytes_FromString(((Request *) self)->request->path);
}

/* Get the data loaded by this request. */
PyObject *
Request_get_data(PyObject *self, PyObject *args, PyObject *kwds)
{
    Request *r = (Request *) self;

    return PyBytes_FromStringAndSize((char *) r->request->udata,
                                              r->request->size);
}

/* Release this request. */
static void
Request_dealloc(PyObject *self)
{
    Request *r = (Request *) self;

    /* Release the wrapped request. */
    if (r->request != NULL) {
        cache_release(r->ustate, r->request);
    }

    /* Release this object. */
    Py_TYPE(self)->tp_free(self);
}

/* Request methods array. */
static PyMethodDef Request_methods[] = {
    {
        "get_filepath",
        (PyCFunction) Request_get_filepath,
        METH_NOARGS,
        "Get the filepath this request loaded."
    },
    {
        "get_data",
        (PyCFunction) Request_get_data,
        METH_NOARGS,
        "Get the data loaded by this request."
    }
};

/* Request type declaration. */
static PyTypeObject PythonRequestType = {
     PyVarObject_HEAD_INIT(NULL, 0)
     .tp_name = "ladcache.Request",
     .tp_doc = PyDoc_STR("File request"),
     .tp_basicsize = sizeof(Request),
     .tp_itemsize = 0,
     .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,

     /* Methods. */
     .tp_init = Generic_init_noop,
     .tp_dealloc = Request_dealloc,
     .tp_methods = Request_methods,
};


/* ------------------------ */
/*    `UserState` METHODS   */
/* ------------------------ */

/* Submit a request for a file to be loaded. */
PyObject *
UserState_submit(PyObject *self, PyObject *args, PyObject *kwds)
{
    int retry = 0; /* Retry submitting until we succeed. */
    char *filepath;

    /* Parse arguments. */
    char *kwlist[] = {"filepath", "retry", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|p", kwlist, &filepath)) {
        PyErr_SetString(PyExc_Exception, "missing/invalid argument");
        return NULL;
    }

    /* Submit, possibly retry until succeeding (i.e., until entry available). */
    int status;
    do {
        status = cache_get_submit(((UserState *) self)->ustate, filepath);
    } while (status == -EAGAIN && retry);
    if (status < 0) {
        PyErr_SetString(PyExc_Exception, strerror(-status));
        return status == -ENOENT ? Py_None : NULL; /* ENOENT is tolerable. */
    }

    Py_INCREF(Py_None);
    return Py_None;
}

/* Reap a request. Waits until a request becomes available, unless WAIT is not
    set, in which case None will be returned if no requests are available. */
PyObject *
UserState_reap(PyObject *self, PyObject *args, PyObject *kwds)
{
    UserState *user_state = (UserState *) self;
    int wait = 1; /* Predicate arguments fill 4 bytes. */

    /* Parse arguments. */
    char *kwlist[] = {"wait", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|p", kwlist, &wait)) {
        PyErr_SetString(PyExc_Exception, "missing/invalid argument");
        return NULL;
    }

    request_t *out;
    int status = wait ? cache_get_reap_wait(user_state->ustate, &out) :
                        cache_get_reap(user_state->ustate, &out);
    if (status == -EAGAIN) {
        Py_INCREF(Py_None);
        return Py_None;
    } else if (status < 0) {
        DEBUG_LOG(SCOPE_INT, LOG_ERROR, "Failed to reap (wait=%d) \"%s\"; %s\n", wait, out->path, strerror(-status));
        PyErr_Format(PyExc_Exception, strerror(-status));
        return NULL;
    }

    /* Allocate and fill the wrapper. */
    Request *request = (Request *) Generic_new(&PythonRequestType, NULL, NULL);
    if (request == NULL) {
        PyErr_SetString(PyExc_Exception, "unable to allocate wrapper");
        cache_release(user_state->ustate, out); /* Don't leak requests. */
        return NULL;
    }
    request->ustate = user_state->ustate;
    request->request = out;

    return (PyObject *) request;
}

/* Get the maximum number of concurrent requests. */
PyObject *
UserState_get_queue_depth(PyObject *self, PyObject *args, PyObject *kwds)
{
    return PyLong_FromUnsignedLong(((UserState *) self)->ustate->queue_depth);
}

/* Get the number in-flight requests. */
PyObject *
UserState_get_in_flight(PyObject *self, PyObject *args, PyObject *kwds)
{
    return PyLong_FromUnsignedLong(((UserState *) self)->ustate->in_flight);
}

/* UserState methods array. */
static PyMethodDef UserState_methods[] = {
    {
        "submit",
        (PyCFunction) UserState_submit,
        METH_VARARGS | METH_KEYWORDS,
        "Submit a request for a file to be loaded."
    },
    {
        "reap",
        (PyCFunction) UserState_reap,
        METH_VARARGS | METH_KEYWORDS,
        "Reap a request."
    },
    {
        "get_queue_depth",
        (PyCFunction) UserState_get_queue_depth,
        METH_NOARGS,
        "Get the maximum number of concurrent requests."
    },
    {
        "get_in_flight",
        (PyCFunction) UserState_get_in_flight,
        METH_NOARGS,
        "Get the number of in-flight requests."
    }
};

/* UserState type declaration. */
static PyTypeObject PythonUserStateType = {
     PyVarObject_HEAD_INIT(NULL, 0)
     .tp_name = "ladcache.UserState",
     .tp_doc = PyDoc_STR("LADCache user context"),
     .tp_basicsize = sizeof(UserState),
     .tp_itemsize = 0,
     .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,

     /* Methods. */
     .tp_init = Generic_init_noop,
     .tp_dealloc = Generic_dealloc,
     .tp_methods = UserState_methods,
};


/* ------------------- */
/*   `Cache` METHODS   */
/* ------------------- */

/* Cache deallocator. */
static void
Cache_dealloc(PyObject *self)
{
    /* Destroy the cache and free the wrapper. */
    cache_destroy(((Cache *) self)->cache);
    Py_TYPE(self)->tp_free(self);
}

/* Cache initializer. */
static int
Cache_init(PyObject *self, PyObject *args, PyObject *kwds)
{
    Cache *c = (Cache *) self;
    size_t capacity;
    int queue_depth;
    int max_unsynced = 1, n_users = 1, debug_limit = 0; /* Default values (optionals). */

    /* Parse arguments. */
    char *kwlist[] = {
        "capacity",
        "queue_depth",
        "max_unsynced",
        "n_users",
        "debug_limit",
        NULL
    };
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ki|iii", kwlist,
                                                &capacity,
                                                &queue_depth,
                                                &max_unsynced,
                                                &n_users,
                                                &debug_limit)) {
        PyErr_SetString(PyExc_Exception, "missing/invalid argument");
        return -1;
    }
    ARG_CHECK(capacity > 0, "capacity must be >= 1 byte", -1);
    ARG_CHECK(queue_depth > 0, "queue_depth must be > 0", -1);
    ARG_CHECK(debug_limit >= 0, "debug_limit must be >= 0", -1);
    ARG_CHECK(max_unsynced >= 0, "max_unsynced must be >= 0", -1);
    ARG_CHECK(n_users >= 0, "n_users must be >= 0", -1);

    /* Allocate the cache_t struct. */
    if ((c->cache = cache_new()) == NULL) {
        PyErr_Format(PyExc_Exception, "Failed to allocate cache_t struct.\n");
        return -1;
    }

    /* Initialize the cache. */
    int status = cache_init(c->cache,
                            capacity,
                            debug_limit,
                            queue_depth,
                            max_unsynced,
                            n_users);
    if (status < 0) {
        PyErr_Format(PyExc_Exception,
                         "Failed to initialize cache; %s\n",
                         strerror(-status));
        return -1;
    }

    return 0;
}

/* Get the wrapped ustate_t for the given user index. */
PyObject *
Cache_get_user_state(PyObject *self, PyObject *args, PyObject *kwds)
{
    Cache *c = (Cache *) self;
    uint32_t index;

    /* Parse the arguments. */
    char *kwlist[] = {"index", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "I", kwlist, &index)) {
        PyErr_SetString(PyExc_Exception, "missing/invalid argument");
        return NULL;
    }

    /* Validate index is within range. */
    ARG_CHECK(index < c->cache->n_users, "invalid user index", NULL);

    /* Allocate and fill the wrapper. */
    UserState *user_state = (UserState *) Generic_new(&PythonUserStateType, NULL, NULL);
    if (user_state == NULL) {
        PyErr_SetString(PyExc_Exception, "unable to allocate wrapper");
        return NULL;
    }
    user_state->ustate = &c->cache->ustates[index];

    return (PyObject *) user_state;
}

/* Spawn the manager, monitor, and registrar threads under a new process. */
PyObject *
Cache_spawn_process(PyObject *self, PyObject *args, PyObject *kwds)
{
    Cache *c = (Cache *) self;

    /* Parent returns immediately. */
    if (fork() != 0) {
        Py_INCREF(Py_None);
        return Py_None;
    }

    /* Child spawns the new threads. We can't use cache_start because we want
       this thread to become the registrar, as opposed to spawning a new thread
       for that and returning here. */
    int status;
    if ((status = manager_spawn(c->cache)) < 0) {
        DEBUG_LOG(SCOPE_INT,
                  LOG_CRITICAL,
                  "Failed to spawn manager; %s\n",
                  strerror(-status));
        return NULL;
    }
    if ((status = monitor_spawn(c->cache)) < 0) {
        DEBUG_LOG(SCOPE_INT,
                  LOG_CRITICAL,
                  "Failed to spawn monitor; %s\n",
                  strerror(-status));
        kill(c->cache->manager_thread, SIGKILL);
        return NULL;
    }

    /* Become the registrar. */
    cache_become_registrar(c->cache);

    /* This should never occur. */
    DEBUG_LOG(SCOPE_INT, LOG_CRITICAL, "Registrar returned unexpectedly.\n");
    kill(c->cache->manager_thread, SIGKILL);
    kill(c->cache->monitor_thread, SIGKILL);
    NOT_REACHED();
    return NULL;
}

/* Cache methods array. */
static PyMethodDef Cache_methods[] = {
    {
        "get_user_state",
        (PyCFunction) Cache_get_user_state,
        METH_VARARGS | METH_KEYWORDS,
        "Get the context for the specified user"
    },
    {
        "spawn_process",
        (PyCFunction) Cache_spawn_process,
        METH_NOARGS,
        "Spawn the manager, monitor, and registrar threads under a new process."
    }
};

/* Cache type declaration. */
static PyTypeObject PythonCacheType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "ladcache.Cache",
    .tp_doc = PyDoc_STR("LADCache cache"),
    .tp_basicsize = sizeof(Cache),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,

    /* Methods. */
    .tp_new = Generic_new,
    .tp_init = Cache_init,
    .tp_dealloc = Cache_dealloc,
    .tp_methods = Cache_methods,
};


/* --------------- */
/*    MODULE INIT  */
/* --------------- */

/* Module definition. */
static struct PyModuleDef LADCacheModule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "ladcache",
    .m_doc = "Locality-aware distributed cache.",
    .m_size = -1,
    .m_methods = NULL,
};

/* Register a Python type with a module. */
#define REGISTER_TYPE(module, name, type_addr)                                 \
    do {                                                                       \
        Py_INCREF(type_addr);                                                  \
        if (PyModule_AddObject(module, name, (PyObject *) type_addr) < 0) {    \
            DEBUG_LOG(SCOPE_INT, LOG_CRITICAL, "Failred to register type.\n"); \
            Py_DECREF(type_addr);                                              \
            Py_DECREF(module);                                                 \
            return NULL;                                                       \
        }                                                                      \
    } while (0)

PyMODINIT_FUNC
PyInit_ladcache(void)
{
    /* Create module. */
    PyObject *module;
    if ((module = PyModule_Create(&LADCacheModule)) == NULL) {
        return NULL;
    }

    /* Ready all types. */
    if (PyType_Ready(&PythonUserStateType) < 0 ||
         PyType_Ready(&PythonCacheType)      < 0 ||
         PyType_Ready(&PythonRequestType)    < 0) {
        return NULL;
    }

    /* Register all types. */
    REGISTER_TYPE(module, "UserState", &PythonUserStateType);
    REGISTER_TYPE(module, "Cache", &PythonCacheType);
    REGISTER_TYPE(module, "Request", &PythonRequestType);

    return module;
}