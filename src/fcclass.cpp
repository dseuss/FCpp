// FIXME Check for memory leaks and corruptions in numpy-like return types
// FIXME Check if RowMajor order is the right thing to do everywhere
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "fcclass.hpp"
// #include "cost.hpp"


namespace py = pybind11;
using namespace pybind11::literals;



FcClassifier::FcClassifier(size_t input_units, const shape_t hidden_units)
: layers(hidden_units.size() + 1)
{
    const auto n_layers = hidden_units.size() + 1;
    if(n_layers < 1) {
        throw std::invalid_argument("Number of layers is too small");
    }

    // Create temporary shapes array to unify loop below
    size_t shapes[n_layers + 1];
    shapes[0] = input_units;
    shapes[n_layers] = 1;
    copy(hidden_units.begin(), hidden_units.end(), shapes + 1);

    // Initialize the weights with zeros
    for (size_t n = 0; n < n_layers; ++n) {
        // +1 to accomodate biases
        layers[n].w = ematrix_t::Zero(shapes[n + 1], shapes[n] + 1);
        layers[n].activation = sigmoid;
    }
}


shape_t FcClassifier::hidden_units () const
{
    shape_t result (layers.size() - 1);
    for (size_t i = 0; i < layers.size() - 1; ++i) {
        result[i] = layers[i].w.rows();
    }
    return result;
}


void FcClassifier::init_random(long seed)
{
    srand(seed);
    for (auto& layer: layers) {
        layer.w = ematrix_t::Random(layer.w.rows(), layer.w.cols());
    }

}


std::vector<ematrix_t> &FcClassifier::get_weights() const
{
    auto result = new std::vector<ematrix_t>(layers.size());
    for (size_t i = 0; i < layers.size(); ++i) {
        (*result)[i] = layers[i].w;
    }
    return *result;
}


void FcClassifier::set_weights(const size_t layer, Eigen::Ref<ematrix_t> weight)
{
    if((weight.rows() != layers[layer].w.rows()) ||
        (weight.cols() != layers[layer].w.cols())) {
        throw std::invalid_argument("Set weight has wrong shape");
    }
    layers[layer].w = weight;
}

// Note that x_in in TensorFlow like with the sample index being the last
// one
evector_t
FcClassifier::predict(const Eigen::Ref<const ematrix_t> x_in) const
{
    ematrix_t x_current = x_in;
    for (auto const& layer: layers) {
        auto w = layer.w.block(0, 1, layer.w.rows(), layer.w.cols() - 1);
        auto b = layer.w.col(0);
        x_current = ((w * x_current).colwise() + b).unaryExpr(layer.activation.f);
    }

    return x_current.row(0);
}

std::vector<ematrix_t>
FcClassifier::back_propagate(const Eigen::Ref<const evector_t> x, const double y) const
{
    auto result = new std::vector<ematrix_t>(layers.size());
    for (size_t i = 0; i < layers.size(); ++i) {
        auto w = layers[i].w;
        (*result)[i] = ematrix_t::Zero(w.rows(), w.cols());
    }
    return *result;
}
