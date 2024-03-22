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

inline PyObject* string2Object(const std::string& str) {
#if PY_MAJOR_VERSION == 2
  return PyString_FromString(str.c_str());
#else
  return PyUnicode_FromString(str.c_str());
#endif
}

typedef struct {
    PyObject_HEAD
    Llm* llm;
} LLM;

static PyObject* PyLLM_new(struct _typeobject *type, PyObject *args, PyObject *kwds) {
    LLM* self = (LLM *)type->tp_alloc(type, 0);
    return (PyObject*)self;
}

static PyObject* Py_str(PyObject *self) {
    char str[50];
    LLM* llm = (LLM*)self;
    sprintf(str, "Llm object: %p", llm->llm);
    return Py_BuildValue("s", str);
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
    return string2Object(res);
}

static PyMethodDef PyLLM_methods[] = {
    {"response", (PyCFunction)PyLLM_response, METH_VARARGS, "response without hsitory."},
    {NULL}  /* Sentinel */
};


static PyObject* PyLLM_get_mgl(LLM *self, void *closure) {
    return PyLong_FromLong(self->llm->max_seq_len_);
}

static int PyLLM_set_mgl(LLM *self, PyObject *value, void *closure) {
    if (self->llm) {
        self->llm->max_seq_len_ = (int)PyLong_AsLong(value);
    }
    return 0;
}

static PyGetSetDef PyLLM_getsetters[] = {
    {"max_gen_len", (getter)PyLLM_get_mgl, (setter)PyLLM_set_mgl, "___max_gen_len___", NULL},
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

static PyObject *py_load(PyObject *self, PyObject *args) {
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
        {"load", py_load, METH_VARARGS},
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