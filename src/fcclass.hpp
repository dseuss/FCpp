#ifndef FCCLASS_HPP_EVZ2HTSU
#define FCCLASS_HPP_EVZ2HTSU

#include <Eigen/Dense>
#include "activation.hpp"


typedef std::vector<size_t> shape_t;

typedef Eigen::Matrix<double,
                      Eigen::Dynamic,
                      Eigen::Dynamic,
                      Eigen::RowMajor> ematrix_t;

typedef Eigen::VectorXd evector_t;

typedef struct NNLayer {
    ActivationFunction activation;
    ematrix_t w;
} NNLayer;


class FcClassifier {
public:
    // A fully connected neural-network classifier.
    // The last layer has a single output unit with sigmoid activation
    FcClassifier(size_t input_units, const shape_t hidden_units);

    size_t input_units() const { return layers[0].w.cols() - 1; }
    size_t hidden_layers() const { return layers.size() - 1; }
    shape_t hidden_units () const;

    void init_random(long seed);

    std::vector<ematrix_t> get_weights() const;
    void set_weights(const size_t layer, Eigen::Ref<ematrix_t> weight);

    // Note that x_in in TensorFlow like with the sample index being the last
    // one
    evector_t predict(const Eigen::Ref<const ematrix_t> x_in) const;
    std::vector<ematrix_t> back_propagate(const Eigen::Ref<const evector_t> x,
                                     const double y) const;

private:
    std::vector<NNLayer> layers;
};


#endif /* end of include guard: FCCLASS_HPP_EVZ2HTSU */
