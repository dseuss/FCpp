#ifndef ACTIVATION_HPP_ZTOI6K93
#define ACTIVATION_HPP_ZTOI6K93

// TODO Can this be expressed as a class? There might be problems since Eigen
//      expects function pointers.
typedef struct ActivationFunction {
  double (*f)(const double);
  double (*df)(const double);
} ActivationFunction;

double sigmoid_f(const double);
double sigmoid_df(const double);
const ActivationFunction sigmoid = {sigmoid_f, sigmoid_df};

double relu_f(const double);
double relu_df(const double);
const ActivationFunction relu = {relu_f, relu_df};

#endif /* end of include guard: ACTIVATION_HPP_ZTOI6K93 */
