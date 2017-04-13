#include <assert.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <armadillo>
#include "interface.hpp"

namespace py = pybind11;
namespace am = arma;

using namespace pybind11::literals;
using namespace std;

typedef vector<size_t> shape_t;

class FCNN {
public:
    // Fully connected neural network
    FCNN(size_t n_inputs, const shape_t neurons)
    : weights(neurons.size())
    {
        const auto n_layers = neurons.size();
        // TODO Proper error handling
        // TODO Get rid of 1-output neuron constraint
        assert(n_layers >= 2);
        assert(*neurons.end() == 1);

        // Initialize the weights with zeros
        weights[0].zeros(neurons[0], n_inputs);
        for (auto n = 1; n < n_layers; ++n) {
            weights[n].zeros(neurons[n], neurons[n - 1]);
        }

    }


    size_t n_inputs() const { return weights[0].n_cols; }
    size_t n_outputs() const { return weights[weights.size() - 1].n_rows; }
    size_t hlayers() const { return weights.size() - 1; }


    shape_t neurons () const
    {
        shape_t result (weights.size());
        for (auto i = 0; i < weights.size(); ++i) {
            result[i] = weights[i].n_rows;
        }
        return result;
    }


private:
    vector<am::Mat<double>> weights;

};


PYBIND11_PLUGIN(fcnn)
{
    py::module m("fcnn");

    py::class_<FCNN>(m, "FCNN")
        .def(py::init<size_t, shape_t>(), "n_inputs"_a, "neurons"_a)
        .def_property_readonly("n_inputs", &FCNN::n_inputs)
        .def_property_readonly("n_outputs", &FCNN::n_outputs)
        .def_property_readonly("hlayers", &FCNN::hlayers)
        .def_property_readonly("neurons", &FCNN::neurons);

    return m.ptr();
}
