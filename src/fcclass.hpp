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


#endif /* end of include guard: FCCLASS_HPP_EVZ2HTSU */
