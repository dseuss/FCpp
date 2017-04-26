// FIXME Check for memory leaks and corruptions in numpy-like return types
// FIXME Check if RowMajor order is the right thing to do everywhere
// FIXME I still don't like the way weights/biasses are handled
// TODO Regularization
// TODO Batch gradient computation

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "fcclass.hpp"
// #include "cost.hpp"


namespace py = pybind11;
using namespace pybind11::literals;


FcClassifier::FcClassifier(size_t input_units, const shape_t hidden_units)
: layers(hidden_units.size() + 1), costfun(cross_entropy)
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
        layers[n].weights = ematrix_t::Zero(shapes[n + 1], shapes[n]);
        layers[n].biases = evector_t::Zero(shapes[n + 1]);
        layers[n].activation = sigmoid;
    }
}


shape_t FcClassifier::hidden_units () const
{
    shape_t result (layers.size() - 1);
    for (size_t i = 0; i < layers.size() - 1; ++i) {
        result[i] = layers[i].biases.size();
    }
    return result;
}


void FcClassifier::init_random(long seed)
{
    srand(seed);
    for (auto& layer: layers) {
        layer.weights = ematrix_t::Random(layer.weights.rows(),
                                          layer.weights.cols());
        layer.biases = evector_t::Random(layer.biases.size());
    }

}


std::vector<weights_biases_t> FcClassifier::get_weights() const
{
    std::vector<weights_biases_t> result(layers.size());
    for (size_t i = 0; i < layers.size(); ++i) {
        result[i] = weights_biases_t(layers[i].weights, layers[i].biases);
    }
    return result;
}


void FcClassifier::set_weights(const size_t layer,
                               const ecref<ematrix_t> weights,
                               const ecref<evector_t> biases)
{
    if ((weights.rows() != layers[layer].weights.rows()) ||
        (weights.cols() != layers[layer].weights.cols())) {
        // FIXME Better error message
        throw std::invalid_argument("Set weights have wrong shape");
    }
    if (biases.size() != layers[layer].biases.size()) {
        throw std::invalid_argument("Set biases have wrong shape");
    }

    layers[layer].weights = weights;
    layers[layer].biases = biases;
}

// Note that x_in in TensorFlow like with the sample index being the last
// one
evector_t FcClassifier::predict(const Eigen::Ref<const ematrix_t> x_in) const
{
    ematrix_t activation = x_in;
    for (auto const& layer: layers) {
        auto lin_activation = (layer.weights * activation).colwise() + layer.biases;
        activation = lin_activation.unaryExpr(layer.activation.f);
    }

    return activation.row(0);
}

double FcClassifier::evaluate(const Eigen::Ref<const ematrix_t> x_in,
                              const Eigen::Ref<const evector_t> y_in) const
{
    if (x_in.cols() != y_in.size()) {
        std::stringstream errmsg;
        errmsg << "Number of samples does not match " << x_in.cols()
            << " != " << y_in.size() << std::endl;
        throw std::invalid_argument(errmsg.str());
    }

    auto y_hat = predict(x_in);
    auto result = 0.0;
    for (auto i = 0; i < y_hat.size(); ++i) {
        result += costfun.f(y_in[i], y_hat[i]);
    }
    return result;
}

std::pair<double, std::vector<weights_biases_t>>
FcClassifier::back_propagate(const Eigen::Ref<const evector_t> x_input,
                             const double y_input) const
{
    // Foward propagate to compute activations
    evector_t activations [layers.size()];
    evector_t lin_activations [layers.size()];

    lin_activations[0] = layers[0].weights * x_input + layers[0].biases;
    activations[0] = lin_activations[0].unaryExpr(layers[0].activation.f);

    for (size_t i = 1; i < layers.size(); ++i) {
        lin_activations[i] = layers[i].weights * activations[i - 1] + layers[i].biases;
        activations[i] = lin_activations[i].unaryExpr(layers[i].activation.f);
    }

    // Back propagate to compute gradients
    std::vector<weights_biases_t> gradients(layers.size());
    evector_t buf (1);
    buf[0] = costfun.d2f(y_input, activations[layers.size() - 1][0]);

    for (size_t i = layers.size() - 1; i > 0; --i) {
        buf.array() *= lin_activations[i].unaryExpr(layers[i].activation.df).array();
        gradients[i] = weights_biases_t(buf * activations[i - 1].transpose(), buf);
        buf = layers[i].weights.transpose() * buf;
    }
    buf = buf.array() * lin_activations[0].unaryExpr(layers[0].activation.df).array();
    gradients[0] = weights_biases_t(buf * x_input.transpose(), buf);

    double cost = costfun.f(y_input, activations[layers.size() - 1][0]);
    const std::pair<double, std::vector<weights_biases_t>> result (cost, gradients);
    return result;
}
