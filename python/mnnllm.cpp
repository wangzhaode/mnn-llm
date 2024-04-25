//
//  mnnllm.cpp
//
//  Created by MNN on 2024/03/22.
//  ZhaodeWang
//

#include <Python.h>
#include <iostream>
#include "llm.hpp"

using namespace std;

// macros
#define def_attr(NAME) \
static PyObject* PyLLM_get_##NAME(LLM *self, void *closure) {\
    return PyLong_FromLong(self->llm->NAME##_);\
}\
static int PyLLM_set_##NAME(LLM *self, PyObject *value, void *closure) {\
    if (self->llm) {\
        self->llm->NAME##_ = PyLong_AsLong(value);\
    }\
    return 0;\
}

#define register_attr(NAME) \
    {#NAME, (getter)PyLLM_get_##NAME, (setter)PyLLM_set_##NAME, "___"#NAME"__", NULL},
// end

// type convert start
inline PyObject* string2Object(const std::string& str) {
#if PY_MAJOR_VERSION == 2
  return PyString_FromString(str.c_str());
#else
  return PyUnicode_FromString(str.c_str());
#endif
}

static inline PyObject* toPyObj(string val) {
    return string2Object(val);
}

static inline PyObject* toPyObj(int val) {
    return PyLong_FromLong(val);
}

template <typename T, PyObject*(*Func)(T)=toPyObj>
static PyObject* toPyObj(vector<T> values) {
    PyObject* obj = PyList_New(values.size());
    for (int i = 0; i < values.size(); i++) {
        PyList_SetItem(obj, i, Func(values[i]));
    }
    return obj;
}

/*
static inline PyObject* toPyArray(MNN::Express::VARP var) {
    auto info = var->getInfo();
    auto shape = info->dim;
    size_t total_length = info->size;
    auto var_ptr = const_cast<void*>(var->readMap<void>());
    std::vector<npy_intp> npy_dims;
    for(const auto dim : shape) {
        npy_dims.push_back(dim);
    }
    // auto data = PyArray_SimpleNewFromData(npy_dims.size(), npy_dims.data(), NPY_FLOAT, ptr);
    auto ndarray = PyArray_SimpleNew(npy_dims.size(), npy_dims.data(), NPY_FLOAT);
    void* npy_ptr = PyArray_DATA((PyArrayObject*)ndarray);
    std::memcpy(npy_ptr, var_ptr, total_length * sizeof(float));
    return (PyObject*)ndarray;
}

static inline PyObject* toPyArray(std::vector<int> vec) {
    npy_intp dims[1] = { static_cast<npy_intp>(vec.size()) };
    auto ndarray = PyArray_SimpleNew(1, dims, NPY_INT);
    void* npy_ptr = PyArray_DATA((PyArrayObject*)ndarray);
    std::memcpy(npy_ptr, vec.data(), vec.size() * sizeof(int));
    return (PyObject*)ndarray;
}
*/

static inline bool isInt(PyObject* obj) {
    return PyLong_Check(obj)
#if PY_MAJOR_VERSION < 3
    || PyInt_Check(obj)
#endif
    ;
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

static inline bool isInts(PyObject* obj) {
    return isInt(obj) || isVec<isInt>(obj);
}

inline int64_t unpackLong(PyObject* obj) {
    int overflow;
    long long value = PyLong_AsLongLongAndOverflow(obj, &overflow);
    return (int64_t)value;
}

static inline int toInt(PyObject* obj) {
    return static_cast<int>(unpackLong(obj));
}

template <typename T, T (*Func)(PyObject*)>
static vector<T> toVec(PyObject* obj) {
    vector<T> values;
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
// type convert end

typedef struct {
    PyObject_HEAD
    Llm* llm;
} LLM;

static PyObject* PyLLM_new(struct _typeobject *type, PyObject *args, PyObject *kwds) {
    LLM* self = (LLM *)type->tp_alloc(type, 0);
    return (PyObject*)self;
}

static PyObject* Py_str(PyObject *self) {
    LLM* llm = (LLM*)self;
    if (!llm) {
        Py_RETURN_NONE;
    }
    return toPyObj(llm->llm->model_name_);
}

static PyObject* PyLLM_load(LLM *self, PyObject *args) {
    const char* model_dir = NULL;
    if (!PyArg_ParseTuple(args, "s", &model_dir)) {
        Py_RETURN_NONE;
    }
    self->llm->load(model_dir);
    Py_RETURN_NONE;
}

static PyObject* PyLLM_generate(LLM *self, PyObject *args) {
    PyObject *input_ids = nullptr;
    if (!PyArg_ParseTuple(args, "O", &input_ids) && isInts(input_ids)) {
        Py_RETURN_NONE;
    }
    auto output_ids = self->llm->generate(toInts(input_ids));
    return toPyObj<int, toPyObj>(output_ids);
}

static PyObject* PyLLM_response(LLM *self, PyObject *args) {
    const char* query = NULL;
    int stream = 0;
    if (!PyArg_ParseTuple(args, "s|p", &query, &stream)) {
        Py_RETURN_NONE;
    }
    LlmStreamBuffer buffer(nullptr);
    std::ostream null_os(&buffer);
    auto res = self->llm->response_nohistory(query, stream ? &std::cout : &null_os);
    return string2Object(res);
}

static PyMethodDef PyLLM_methods[] = {
    {"load", (PyCFunction)PyLLM_load, METH_VARARGS, "load model from `dir`."},
    {"generate", (PyCFunction)PyLLM_generate, METH_VARARGS, "generate `output_ids` by `input_ids`."},
    {"response", (PyCFunction)PyLLM_response, METH_VARARGS, "response `query` without hsitory."},
    {NULL}  /* Sentinel */
};

def_attr(backend_type)
def_attr(thread_num)
def_attr(low_precision)
def_attr(chatml)
def_attr(max_new_tokens)

static PyGetSetDef PyLLM_getsetters[] = {
    register_attr(backend_type)
    register_attr(thread_num)
    register_attr(low_precision)
    register_attr(chatml)
    register_attr(max_new_tokens)
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
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
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

static PyObject* py_create(PyObject *self, PyObject *args) {
    if (!PyTuple_Size(args)) {
        return NULL;
    }
    const char* model_dir = NULL;
    const char* model_type = "auto";
    if (!PyArg_ParseTuple(args, "s|s", &model_dir, &model_type)) {
        return NULL;
    }
    LLM *llm = (LLM *)PyObject_Call((PyObject*)&PyLLM, PyTuple_New(0), NULL);
    if (!llm) {
        return NULL;
    }
    llm->llm = Llm::createLLM(model_dir, model_type);
    // llm->llm->load(model_dir);
    return (PyObject*)llm;
}

static PyMethodDef Methods[] = {
    {"create", py_create, METH_VARARGS},
    {NULL, NULL}
};

static struct PyModuleDef mnnllmModule = {
        PyModuleDef_HEAD_INIT,
        "cmnnllm", /*module name*/
        "mnnllm cpython module.", /* module documentation, may be NULL */
        -1, /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
        Methods
};

static void def(PyObject* m, PyMethodDef* method) {
    PyModule_AddObject(m, method->ml_name, PyCFunction_New(method, 0));
}

PyMODINIT_FUNC PyInit_cmnnllm(void) {
    if (PyType_Ready(&PyLLM) < 0) {
        PyErr_SetString(PyExc_Exception, "init LLM: PyType_Ready PyLLM failed.");
        return NULL;
    }
    PyObject *m = PyModule_Create(&mnnllmModule);
    // _import_array();
    PyModule_AddObject(m, "LLM", (PyObject *)&PyLLM);
    def(m, &Methods[0]);
    return m;
}