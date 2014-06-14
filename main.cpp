#include <iostream>
#include <time.h>

#include <autodiff/autodiff.hpp>
#include <scalar/functions.hpp>
#include <scalar/operators.hpp>

#include <tests/validate_exceptions.hpp>

using namespace nomad;

template <typename T, int N>
struct funnel_func {
  T operator()(const Eigen::VectorXd& x) const {
    
    T v = x[0];
    
    T y[N];
    for (int n = 0; n < N; ++n)
      y[n] = x[n + 1];
    
    T sum_x2 = 0.0;
    
    for (int n = 0; n < N; ++n)
      sum_x2 += square(y[n]);
    
    T p1 = 0.5 * N * v;
    T p2 = 0.5 * sum_x2 * exp(-v);
    T p3 = 0.5 * square(v) / 9.0;
    
    return p1 + p2 + p3;
    
  }
};

inline double elapsed_secs(const clock_t& start) {
  return (double)(clock() - start) / CLOCKS_PER_SEC;
}

void time_funnel() {

  clock_t start;
  double deltaT;
  
  int n_calls = 5000;
  int n_calls_times_100 = 100 * n_calls;
  
  Eigen::VectorXd x = Eigen::VectorXd::Ones(101);
  Eigen::VectorXd v = Eigen::VectorXd::Ones(101);
  Eigen::MatrixXd M = Eigen::MatrixXd::Identity(101, 101);
  
  double f;
  Eigen::VectorXd grad(x.size());
  Eigen::MatrixXd H(x.size(), x.size());
  Eigen::VectorXd grad_m_times_h(x.size());
  
  // First-order timing
  funnel_func<var<1U>, 100> first_order_funnel;

  start = clock();
  for (int n = 0; n < n_calls_times_100; ++n) {
    gradient(first_order_funnel, x, f, grad);
  }
  deltaT = elapsed_secs(start);
  
  std::cout << n_calls_times_100 << " gradients took " << deltaT << " seconds" << std::endl;
  
  // Second-order timing
  funnel_func<var<2U>, 100> second_order_funnel;
  
  start = clock();
  for (int n = 0; n < n_calls; ++n)
    hessian(second_order_funnel, x, f, grad, H);
  deltaT = elapsed_secs(start);
  
  std::cout << n_calls << " hessians took " << deltaT << " seconds"
            << std::endl;
  
  // Third-order timing
  funnel_func<var<3U>, 100> third_order_funnel;
  
  start = clock();
  for (int n = 0; n < n_calls; ++n)
    grad_trace_matrix_times_hessian(third_order_funnel, x, M, f,
                                    grad, H, grad_m_times_h);
  deltaT = elapsed_secs(start);
  
  std::cout << n_calls << " grad-matrix-times-hessians took "
            << deltaT << " seconds" << std::endl;
  
}

void validate_funnel() {
  
  Eigen::VectorXd x = Eigen::VectorXd::Ones(6);
  Eigen::VectorXd v = Eigen::VectorXd::Ones(6);
  Eigen::MatrixXd M = Eigen::MatrixXd::Identity(6, 6);
  
  x *= 0.345;

  // First-order
  funnel_func<var<1U>, 5> first_order_func;
  test_gradient(first_order_func, x);
  test_gradient_dot_vector(first_order_func, x, v);
  
  // Second-order
  funnel_func<var<2U>, 5> second_order_func;
  test_hessian(second_order_func, x);
  test_hessian_dot_vector(second_order_func, x, v);
  test_trace_matrix_times_hessian(second_order_func, x, M);
   
  // Third-order
  funnel_func<var<3U>, 5> third_order_func;
  test_grad_hessian(third_order_func, x);
  test_grad_trace_matrix_times_hessian(third_order_func, x, M);

}

int main(int argc, const char * argv[]) {
  //validate_exceptions();
  //validate_funnel();
  time_funnel();
  return 0;
}



