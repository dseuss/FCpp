#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "fcclass.hpp"

namespace py = pybind11;
using namespace pybind11::literals;


PYBIND11_PLUGIN(fcclass)
{
    py::module m("fcclass");

    py::class_<FcClassifier>(m, "FcClassifier")
        .def(py::init<size_t, shape_t>(), "input_units"_a, "hidden_units"_a)
        .def_property_readonly("input_units", &FcClassifier::input_units)
        .def_property_readonly("hidden_layers", &FcClassifier::hidden_layers)
        .def_property_readonly("hidden_units", &FcClassifier::hidden_units)
        .def("init_random", &FcClassifier::init_random, "seed"_a=0)
        .def("get_weights", &FcClassifier::get_weights,
             py::return_value_policy::copy)
        .def("set_weights", &FcClassifier::set_weights, "layer"_a, "weight"_a)
        .def("predict", &FcClassifier::predict, "x_in"_a)
        .def("back_propagate", &FcClassifier::back_propagate,
             py::return_value_policy::copy, "x"_a, "y"_a);

    return m.ptr();
}
