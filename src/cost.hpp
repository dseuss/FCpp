#ifndef COST_HPP_8R4FCWZV
#define COST_HPP_8R4FCWZV

typedef struct CostFunction {
  double (*f)(const double, const double);
  double (*d1f)(const double, const double);
  double (*d2f)(const double, const double);
} CostFunction;

/*********************************
 *  cross entropy cost function  *
 *********************************/
double cross_entropy_f(const double p, const double q);
double cross_entropy_d1f(const double, const double q);
double cross_entropy_d2f(const double p, const double q);
const CostFunction cross_entropy = {cross_entropy_f, cross_entropy_d1f,
                                    cross_entropy_d2f};

#endif /* end of include guard: COST_HPP_8R4FCWZV */
