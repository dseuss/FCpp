#ifndef FCCLASS_HPP_EVZ2HTSU
#define FCCLASS_HPP_EVZ2HTSU

#include <Eigen/Dense>
#include <vector>

#include "activation.hpp"
#include "cost.hpp"

typedef std::vector<size_t> shape_t;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    ematrix_t;

typedef Eigen::VectorXd evector_t;
typedef std::pair<ematrix_t, evector_t> weights_biases_t;

template <typename T>
using ecref = Eigen::Ref<const T>;

typedef struct NNLayer {
  ActivationFunction activation;
  ematrix_t weights;
  evector_t biases;
} NNLayer;

class FcClassifier {
 public:
  // A fully connected neural-network classifier.
  // The last layer has a single output unit with sigmoid activation
  FcClassifier(size_t input_units, const shape_t hidden_units);

  size_t input_units() const { return layers[0].weights.cols(); }
  size_t hidden_layers() const { return layers.size() - 1; }
  shape_t hidden_units() const;

  void init_random(long seed);

  std::vector<weights_biases_t> get_weights() const;
  void set_weights(const size_t layer, const ecref<ematrix_t> weight,
                   const ecref<evector_t> bias);

  // Note that x_in in TensorFlow like with the sample index being the last
  // one
  evector_t predict(const ecref<ematrix_t> x_in) const;
  double evaluate(const ecref<ematrix_t> x_in,
                  const ecref<evector_t> y_in) const;
  std::pair<double, std::vector<weights_biases_t>> back_propagate(
      const ecref<evector_t> x, const double y) const;

  double train(const std::vector<ecref<evector_t>> x_input,
               const std::vector<double> y_input, const double learning_rate,
               const unsigned int nr_epochs, const unsigned int batch_size,
               const unsigned int seed);

 private:
  std::vector<NNLayer> layers;
  const CostFunction costfun;
  double train_epoch(const std::vector<ecref<evector_t>> x_input,
                     const std::vector<double> y_input,
                     const std::vector<size_t> indices,
                     const double learning_rate, const unsigned int batch_size);
};

#endif /* end of include guard: FCCLASS_HPP_EVZ2HTSU */
