#include <assert.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <armadillo>

#include "activation.hpp"


namespace py = pybind11;
namespace am = arma;
using namespace pybind11::literals;
using namespace std;


typedef vector<size_t> shape_t;


typedef struct NNLayer {
    ActivationFunction activation;
    am::Mat<double> w;
    am::Col<double> b;
} NNLayer;


class FCNN {
public:
    // Fully connected neural network
    FCNN(size_t n_inputs, const shape_t neurons)
    : layers(neurons.size())
    {
        const auto n_layers = neurons.size();
        // TODO Proper error handling
        // TODO Get rid of 1-output neuron constraint
        assert(n_layers >= 2);
        assert(*neurons.end() == 1);

        // Create temporary shapes array to unify loop below
        size_t shapes[neurons.size() + 1];
        shapes[0] = n_inputs;
        copy(neurons.begin(), neurons.end(), shapes + 1);

        // Initialize the weights with zeros
        for (size_t n = 0; n < n_layers; ++n) {
            layers[n].w.zeros(shapes[n + 1], shapes[n]);
            layers[n].b.zeros(shapes[n + 1]);
            layers[n].activation = sigmoid;
        }
    }


    size_t n_inputs() const { return layers[0].w.n_cols; }
    size_t n_outputs() const { return layers[layers.size() - 1].w.n_rows; }
    size_t hlayers() const { return layers.size() - 1; }


    shape_t neurons () const
    {
        shape_t result (layers.size());
        for (size_t i = 0; i < layers.size(); ++i) {
            result[i] = layers[i].w.n_rows;
        }
        return result;
    }


private:
    vector<NNLayer> layers;
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
