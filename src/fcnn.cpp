#include <assert.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>

#include "activation.hpp"


namespace py = pybind11;
using namespace pybind11::literals;
using namespace std;


typedef vector<size_t> shape_t;
typedef Eigen::Matrix<double,
                      Eigen::Dynamic,
                      Eigen::Dynamic,
                      Eigen::RowMajor> matrix_t;

typedef struct NNLayer {
    ActivationFunction activation;
    matrix_t w;
} NNLayer;


class FCNN {
public:
    // Fully connected neural network
    FCNN(size_t n_inputs, const shape_t neurons)
    : layers(neurons.size())
    {
        const auto n_layers = neurons.size();
        if(n_layers < 1) {
            throw invalid_argument("Number of layers is too small");
        }

        // Create temporary shapes array to unify loop below
        size_t shapes[neurons.size() + 1];
        shapes[0] = n_inputs;
        copy(neurons.begin(), neurons.end(), shapes + 1);

        // Initialize the weights with zeros
        for (size_t n = 0; n < n_layers; ++n) {
            // +1 to accomodate biases
            layers[n].w = matrix_t::Zero(shapes[n + 1], shapes[n] + 1);
            layers[n].activation = sigmoid;
        }
    }


    size_t n_inputs() const { return layers[0].w.cols() - 1; }
    size_t n_outputs() const { return layers[layers.size() - 1].w.rows(); }
    size_t hlayers() const { return layers.size() - 1; }


    shape_t neurons () const
    {
        shape_t result (layers.size());
        for (size_t i = 0; i < layers.size(); ++i) {
            result[i] = layers[i].w.rows();
        }
        return result;
    }


    matrix_t &get_weights(const size_t layer)
    {
        return layers[layer].w;
    }


    void set_weights(const size_t layer, Eigen::Ref<matrix_t> weight)
    {
        if((weight.rows() != layers[layer].w.rows()) ||
           (weight.cols() != layers[layer].w.cols())) {
            throw invalid_argument("Set weight has wrong shape");
        }
        layers[layer].w = weight;
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
        .def_property_readonly("neurons", &FCNN::neurons)
        .def("get_weights", &FCNN::get_weights, "layer"_a)
        .def("set_weights", &FCNN::set_weights, "layer"_a, "weight"_a);

    return m.ptr();
}
