//
//  mnnllm.cpp
//
//  Created by MNN on 2024/03/22.
//  ZhaodeWang
//

#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>

#include <iostream>
#include "llm.hpp"

using namespace std;

// type functions
inline PyObject* string2Object(const std::string& str) {
#if PY_MAJOR_VERSION == 2
    return PyString_FromString(str.c_str());
#else
    return PyUnicode_FromString(str.c_str());
#endif
}

inline int64_t unpackLong(PyObject* obj) {
    int overflow;
    long long value = PyLong_AsLongLongAndOverflow(obj, &overflow);
    return (int64_t)value;
}

template <bool (*Func)(PyObject*)>
static bool isVec(PyObject* obj) {
    if (PyTuple_Check(obj)) {
        if (PyTuple_Size(obj) > 0) {
            return Func(PyTuple_GetItem(obj, 0));
        } else return true;
    } else if (PyList_Check(obj)) {
        if (PyList_Size(obj) > 0) {
            return Func(PyList_GetItem(obj, 0));
        } else return true;
    }
    return false;
}

static inline bool isInt(PyObject* obj) {
    return PyLong_Check(obj)
#if PY_MAJOR_VERSION < 3
    || PyInt_Check(obj)
#endif
    ;
}

static inline bool isInts(PyObject* obj) {
    return isInt(obj) || isVec<isInt>(obj);
}

static inline int toInt(PyObject* obj) {
    return static_cast<int>(unpackLong(obj));
}

template <typename T, T (*Func)(PyObject*)>
static std::vector<T> toVec(PyObject* obj) {
    std::vector<T> values;
    if (PyTuple_Check(obj)) {
        size_t size = PyTuple_Size(obj);
        values.resize(size);
        for (int i = 0; i < size; i++) {
            values[i] = Func(PyTuple_GetItem(obj, i));
        }
        return values;
    }
    if (PyList_Check(obj)) {
        size_t size = PyList_Size(obj);
        values.resize(size);
        for (int i = 0; i < size; i++) {
            values[i] = Func(PyList_GetItem(obj, i));
        }
        return values;
    }
    values.push_back(Func(obj));
    return values;
}

static inline std::vector<int> toInts(PyObject* obj) {
    if (isInt(obj)) { return { toInt(obj) }; }
    return toVec<int, toInt>(obj);
}

static inline PyObject* toPyObj(string val) {
    return string2Object(val);
}

static inline PyObject* toPyArray(MNN::Express::VARP var) {
    auto info = var->getInfo();
    auto shape = info->dim;
    int64_t total_length = info->size;
    auto ptr = const_cast<void*>(var->readMap<void>());
    std::vector<npy_intp> npy_dims;
    for(const auto dim : shape) {
        npy_dims.push_back(dim);
    }
    auto data = PyArray_SimpleNewFromData(npy_dims.size(), npy_dims.data(), NPY_FLOAT, ptr);
    var->unMap();
    return (PyObject*)data;
}
// end

typedef struct {
    PyObject_HEAD
    Llm* llm;
} LLM;

static PyObject* PyLLM_new(struct _typeobject *type, PyObject *args, PyObject *kwds) {
    LLM* self = (LLM *)type->tp_alloc(type, 0);
    return (PyObject*)self;
}

// LLM's functions
static PyObject* Py_str(PyObject *self) {
    LLM* llm = (LLM*)self;
    std::string str = "Llm: " + llm->llm->model_name_;
    return toPyObj(str);
}

static PyObject* PyLLM_load(LLM *self, PyObject *args) {
    const char* dir = NULL;
    if (!PyArg_ParseTuple(args, "s", &dir)) {
        return NULL;
    }
    self->llm->load(dir);
    Py_RETURN_NONE;
}

static PyObject* PyLLM_response(LLM *self, PyObject *args) {
    const char* query = NULL;
    int stream = 0;
    if (!PyArg_ParseTuple(args, "s|p", &query, &stream)) {
        return NULL;
    }
    LlmStreamBuffer buffer(nullptr);
    std::ostream null_os(&buffer);
    auto res = self->llm->response_nohistory(query, stream ? &std::cout : &null_os);
    return toPyObj(res);
}

static PyObject* PyLLM_logits(LLM *self, PyObject *args) {
    PyObject *input_ids = nullptr;
    if (!PyArg_ParseTuple(args, "O", &input_ids) && isInts(input_ids)) {
        return NULL;
    }
    auto logits = self->llm->logits(toInts(input_ids));
    return toPyArray(logits);
}

static PyMethodDef PyLLM_methods[] = {
    {"load", (PyCFunction)PyLLM_load, METH_VARARGS, "load model from `dir`."},
    {"logits", (PyCFunction)PyLLM_logits, METH_VARARGS, "get logits of `input_ids`."},
    {"response", (PyCFunction)PyLLM_response, METH_VARARGS, "response `query` without hsitory."},
    {NULL}  /* Sentinel */
};

// LLM's attributes
static PyObject* PyLLM_get_backend_type(LLM *self, void *closure) {
    return PyLong_FromLong(self->llm->backend_type_);
}

static int PyLLM_set_backend_type(LLM *self, PyObject *value, void *closure) {
    if (self->llm) {
        self->llm->backend_type_ = (int)PyLong_AsLong(value);
    }
    return 0;
}

static PyObject* PyLLM_get_thread_num(LLM *self, void *closure) {
    return PyLong_FromLong(self->llm->thread_num_);
}

static int PyLLM_set_thread_num(LLM *self, PyObject *value, void *closure) {
    if (self->llm) {
        self->llm->thread_num_ = (int)PyLong_AsLong(value);
    }
    return 0;
}

static PyObject* PyLLM_get_low_precision(LLM *self, void *closure) {
    return PyLong_FromLong(self->llm->low_precision_);
}

static int PyLLM_set_low_precision(LLM *self, PyObject *value, void *closure) {
    if (self->llm) {
        self->llm->low_precision_ = PyLong_AsLong(value);
    }
    return 0;
}

static PyObject* PyLLM_get_chatml(LLM *self, void *closure) {
    return PyLong_FromLong(self->llm->chatml_);
}

static int PyLLM_set_chatml(LLM *self, PyObject *value, void *closure) {
    if (self->llm) {
        self->llm->chatml_ = PyLong_AsLong(value);
    }
    return 0;
}

static PyObject* PyLLM_get_max_new_tokens(LLM *self, void *closure) {
    return PyLong_FromLong(self->llm->max_new_tokens_);
}

static int PyLLM_set_max_new_tokens(LLM *self, PyObject *value, void *closure) {
    if (self->llm) {
        self->llm->max_new_tokens_ = (int)PyLong_AsLong(value);
    }
    return 0;
}

static PyGetSetDef PyLLM_getsetters[] = {
    {"backend_type", (getter)PyLLM_get_backend_type, (setter)PyLLM_set_backend_type, "___backend_type___", NULL},
    {"thread_num", (getter)PyLLM_get_thread_num, (setter)PyLLM_set_thread_num, "___thread_num___", NULL},
    {"low_precision", (getter)PyLLM_get_low_precision, (setter)PyLLM_set_low_precision, "___low_precision___", NULL},
    {"chatml", (getter)PyLLM_get_chatml, (setter)PyLLM_set_chatml, "___chatml___", NULL},
    {"max_new_tokens", (getter)PyLLM_get_max_new_tokens, (setter)PyLLM_set_max_new_tokens, "___max_new_tokens___", NULL},
    {NULL}  /* Sentinel */
};

static PyTypeObject PyLLM = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "LLM",                                    /*tp_name*/
    sizeof(LLM),                              /*tp_basicsize*/
    0,                                        /*tp_itemsize*/
    0,                                        /*tp_dealloc*/
    0,                                        /*tp_print*/
    0,                                        /*tp_getattr*/
    0,                                        /*tp_setattr*/
    0,                                        /*tp_compare*/
    Py_str,                                   /*tp_repr*/
    0,                                        /*tp_as_number*/
    0,                                        /*tp_as_sequence*/
    0,                                        /*tp_as_mapping*/
    0,                                        /*tp_hash */
    0,                                        /*tp_call*/
    Py_str,                                   /*tp_str*/
    0,                                        /*tp_getattro*/
    0,                                        /*tp_setattro*/
    0,                                        /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE, /*tp_flags*/
    "LLM is mnn-llm's `Llm` python wrapper",  /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                                        /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    PyLLM_methods,                            /* tp_methods */
    0,                                        /* tp_members */
    PyLLM_getsetters,                         /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    0,                                        /* tp_init */
    0,                                        /* tp_alloc */
    PyLLM_new,                                /* tp_new */
};

static PyObject *py_create(PyObject *self, PyObject *args) {
    if (!PyTuple_Size(args)) {
        return NULL;
    }
    const char *model_dir = NULL;
    const char* model_type = "auto";
    if (!PyArg_ParseTuple(args, "s|s", &model_dir, &model_type)) {
        return NULL;
    }
    LLM *llm = (LLM *)PyObject_Call((PyObject*)&PyLLM, PyTuple_New(0), NULL);
    if (!llm) {
        return NULL;
    }
    llm->llm = Llm::createLLM(model_dir, model_type);
    llm->llm->load(model_dir);
    return (PyObject*)llm;
}

static PyMethodDef Methods[] = {
    {"create", py_create, METH_VARARGS},
    {NULL, NULL}
};

static struct PyModuleDef mnnllmModule = {
    PyModuleDef_HEAD_INIT,
    "cmnnllm", /*module name*/
    "", /* module documentation, may be NULL */
    -1, /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    Methods
};

static void def(PyObject* m, PyMethodDef* method) {
    PyModule_AddObject(m, method->ml_name, PyCFunction_New(method, 0));
}

PyMODINIT_FUNC PyInit_cmnnllm(void) {
    PyObject *m = PyModule_Create(&mnnllmModule);
    if (PyType_Ready(&PyLLM) < 0) {
        PyErr_SetString(PyExc_Exception, "init LLM: PyType_Ready PyLLM failed");
    }
    PyModule_AddObject(m, "LLM", (PyObject *)&PyLLM);
    def(m, &Methods[0]);
    return m;
}