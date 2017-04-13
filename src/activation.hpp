#ifndef ACTIVATION_HPP_ZTOI6K93
#define ACTIVATION_HPP_ZTOI6K93

#include <cmath>

typedef struct ActivationFunction {
    double (*f)(double);
    double (*df)(double);
} ActivationFunction;


double sigmoid_f(const double x)
{
    return 1 / (1 + exp(x));
}
double sigmoid_df(const double x)
{
    return sigmoid_f(x) * (1 - sigmoid_f(x));
}
const ActivationFunction sigmoid = { sigmoid_f, sigmoid_df };

#endif /* end of include guard: ACTIVATION_HPP_ZTOI6K93 */

