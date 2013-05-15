#include <Python.h>
#include <numpy/arrayobject.h>
#include <Eigen/Core>
#include "include/nca.hpp"

static PyObject *nca_nca(PyObject *self, PyObject *args)
{
  PyArrayObject *py_input;
  PyObject *py_labels;

  if(!PyArg_ParseTuple(args, "OO", &py_input, &py_labels))
    return NULL;

  if(!PyArray_Check(py_input)) {
    PyErr_SetString(PyExc_ValueError, "input must be a numpy array");
    return NULL;
  }
  if(!PySequence_Check(py_labels)) {
    PyErr_SetString(PyExc_ValueError, "labels must be a sequence");
    return NULL;
  }

  int nrows = PyArray_DIM(py_input, 0);
  int ncols = PyArray_DIM(py_input, 1);

  if(PySequence_Size(py_labels) != nrows) {
    PyErr_SetString(PyExc_ValueError, "there must be as many labels as "
                    "there are rows in the input array");
    return NULL;
  }

  std::vector<Eigen::VectorXd> input;
  std::vector<std::string> labels;
  
  for(int i = 0; i < nrows; i ++) {
    Eigen::VectorXd row(ncols);
    for(int j = 0; j < ncols; j ++) {
      row[j] = *(double *)PyArray_GETPTR2(py_input, i, j);
    }

    input.push_back(row);

    PyObject *py_label = PySequence_GetItem(py_labels, i);
    PyObject *label_repr = PyObject_Repr(py_label);
    std::string label(PyString_AsString(label_repr));
    Py_DECREF(label_repr);
    Py_DECREF(py_label);
    labels.push_back(label);
  }

  int iterations = 100000;
  double learning_rate = 0.01;

  Eigen::MatrixXd transform =
    neighborhood_components_analysis(input, labels, scaling_matrix(input),
                                     iterations, learning_rate);

  int dim[2] = {ncols, ncols};
  PyObject *py_transform = PyArray_EMPTY(2, dim, NPY_DOUBLE, 0);

  for(int i = 0; i < nrows; i ++) {
    for(int j = 0; j < ncols; j ++) {
      double *ptr = (double *)PyArray_GETPTR2(py_transform, i, j);
      *ptr = transform(i, j);
    }
  }

  return py_transform;
}

extern "C" {

  static PyMethodDef nca_methods[] = {
    {"nca", nca_nca, METH_VARARGS, "Return the NCA mahalanobis matrix"},
    {NULL, NULL, 0, NULL}
  };

  PyMODINIT_FUNC initnca(void)
  {
    import_array();
    Py_InitModule("nca", nca_methods);
  }
}
