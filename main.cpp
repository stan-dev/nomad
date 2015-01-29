#include <iostream>
#include <time.h>

#include <src/autodiff/autodiff.hpp>
#include <src/scalar/functions.hpp>
#include <src/scalar/operators.hpp>
#include <src/matrix/functions.hpp>

using namespace nomad;

double elapsed_secs(const clock_t& start);
void validate_funnel();
void time_funnel();
void validate_matrix();
void time_matrix();
void validate_dot();
void time_dot();

template <typename T, int N>
class funnel_func: public base_functor<T> {
public:
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

template <typename T, int N>
class debug_func: public base_functor<T> {
public:
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
    
    return p1 + p2 + p3 + inv_sqrt(v);
    
  }
};

template <typename T>
class f_matrix: public base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    
    eigen_idx_t N = static_cast<eigen_idx_t>(std::sqrt(x.size()));
    
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> M(N, N);
    
    int k = 0;
    for (eigen_idx_t i = 0; i < N; ++i)
      for (eigen_idx_t j = 0; j < N; ++j)
        M(i, j) = x(k++);
    
    return sum(multiply(M, M.transpose()));
    
  }
};

/*
template <typename T>
class f_dot: public base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    
    eigen_idx_t N = x.size() / 2;
    
    Eigen::Matrix<T, Eigen::Dynamic, 1> v1(N);
    Eigen::Matrix<T, Eigen::Dynamic, 1> v2(N);
    
    for (eigen_idx_t n = 0; n < N; ++n) v1(n) = x(n);
    for (eigen_idx_t n = 0; n < N; ++n) v2(n) = x(N + n);
    
    return dot(v1, v2);
    
  }
};
*/

inline double elapsed_secs(const clock_t& start) {
  return static_cast<double>(clock() - start) / CLOCKS_PER_SEC;
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
  funnel_func<var1, 100> first_order_funnel;

  start = clock();
  for (int n = 0; n < n_calls_times_100; ++n) {
    gradient(first_order_funnel, x, f, grad);
  }
  deltaT = elapsed_secs(start);
  
  std::cout << n_calls_times_100 << " gradients took " << deltaT << " seconds" << std::endl;
  
  // Second-order timing
  funnel_func<var2, 100> second_order_funnel;
  
  start = clock();
  for (int n = 0; n < n_calls; ++n)
    hessian(second_order_funnel, x, f, grad, H);
  deltaT = elapsed_secs(start);
  
  std::cout << n_calls << " hessians took " << deltaT << " seconds"
            << std::endl;
  
  // Third-order timing
  funnel_func<var3, 100> third_order_funnel;
  
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
  funnel_func<wild_var1, 5> first_order_func;
  test_gradient(first_order_func, x);
  test_gradient_dot_vector(first_order_func, x, v);
  
  // Second-order
  funnel_func<wild_var2, 5> second_order_func;
  test_hessian(second_order_func, x);
  test_hessian_dot_vector(second_order_func, x, v);
  test_trace_matrix_times_hessian(second_order_func, x, M);
   
  // Third-order
  funnel_func<wild_var3, 5> third_order_func;
  test_grad_hessian(third_order_func, x);
  test_grad_trace_matrix_times_hessian(third_order_func, x, M);

}

/*
void validate_matrix() {

  f_matrix<var<1U> > first_order_func;
  f_matrix<var<2U> > second_order_func;
  f_matrix<var<3U> > third_order_func;
  
  const int N = 10;
  Eigen::VectorXd x(N * N);
  for (int n = 0; n < N * N; ++n)
    x(n) = static_cast<double>(n % N);
  
  Eigen::VectorXd v = Eigen::VectorXd::Ones(N * N);
  Eigen::MatrixXd M = Eigen::MatrixXd::Ones(N * N, N * N);
  
  // First-order
  test_gradient(first_order_func, x);
  test_gradient_dot_vector(first_order_func, x, v);
  
  // Second-order
  test_hessian(second_order_func, x);
  test_hessian_dot_vector(second_order_func, x, v);
  test_trace_matrix_times_hessian(second_order_func, x, M);
  
  // Third-order
  //test_grad_hessian(third_order_func, x);
  test_grad_trace_matrix_times_hessian(third_order_func, x, M);
}

void time_matrix() {

  // Functions
  f_matrix<var<1U> > first_order_func;
  f_matrix<var<2U> > second_order_func;
  f_matrix<var<3U> > third_order_func;
  
  const int N = 10;
  Eigen::VectorXd x(N * N);
  for (int n = 0; n < N * N; ++n)
    x(n) = static_cast<double>(n % N);
  
  Eigen::MatrixXd M = Eigen::MatrixXd::Ones(N * N, N * N);
  
  clock_t start;
  double deltaT;
  
  int n_calls = 5000;
  
  double f;
  Eigen::VectorXd grad(x.size());
  Eigen::MatrixXd H(x.size(), x.size());
  Eigen::VectorXd grad_m_times_h(x.size());
  
  // First-order timing
  start = clock();
  for (int n = 0; n < 100 * n_calls; ++n)
    gradient(first_order_func, x, f, grad);
  deltaT = elapsed_secs(start);
  
  std::cout << 100 * n_calls << " gradients took " << deltaT << " seconds"
  << std::endl;
  
  // Second-order timing
  start = clock();
  for (int n = 0; n < n_calls; ++n)
    hessian(second_order_func, x, f, grad, H);
  deltaT = elapsed_secs(start);
  
  std::cout << n_calls << " hessians took " << deltaT << " seconds" << std::endl;
  
  // Third-order timing
  start = clock();
  for (int n = 0; n < n_calls; ++n)
    grad_trace_matrix_times_hessian(third_order_func, x, M, f, grad, H, grad_m_times_h);
  deltaT = elapsed_secs(start);
  
  std::cout << n_calls << " grad-matrix-times-hessians took " << deltaT << " seconds" << std::endl;
  
}

void validate_dot() {
  
  const int N = 5;
  Eigen::VectorXd x(2 * N);
  for (int n = 0; n < 2 * N; ++n)
    x(n) = static_cast<double>(n % N);
  
  Eigen::VectorXd v = Eigen::VectorXd::Ones(2 * N);
  Eigen::MatrixXd M = Eigen::MatrixXd::Ones(2 * N, 2 * N);
  
  // First-order
  f_dot<var<1U> > first_order_func;
  test_gradient(first_order_func, x);
  test_gradient_dot_vector(first_order_func, x, v);
  
  // Second-order
  f_dot<var<2U> > second_order_func;
  test_hessian(second_order_func, x);
  test_hessian_dot_vector(second_order_func, x, v);
  test_trace_matrix_times_hessian(second_order_func, x, M);
  
  // Third-order
  f_dot<var<3U> > third_order_func;
  test_grad_hessian(third_order_func, x);
  test_grad_trace_matrix_times_hessian(third_order_func, x, M);
  
}

void time_dot() {
  
  clock_t start;
  clock_t end;
  double deltaT;
  
  int n_calls = 5000;
  
  const int N = 100;
  Eigen::VectorXd x(2 * N);
  for (int n = 0; n < 2 * N; ++n)
    x(n) = static_cast<double>(n % N);
  
  Eigen::VectorXd v = Eigen::VectorXd::Ones(2 * N);
  Eigen::MatrixXd M = Eigen::MatrixXd::Ones(2 * N, 2 * N);
  
  double f;
  Eigen::VectorXd grad(x.size());
  Eigen::MatrixXd H(x.size(), x.size());
  Eigen::VectorXd grad_m_times_h(x.size());
  
  // First-order timing
  f_dot<var<1U> > first_order_func;
  
  start = clock();
  for (int n = 0; n < 100 * n_calls; ++n)
    gradient(first_order_func, x, f, grad);
  end = clock();
  
  deltaT = (double)(end - start) / CLOCKS_PER_SEC;
  
  std::cout << 100 * n_calls << " gradients took " << deltaT << " seconds" << std::endl;
  
  // Second-order timing
  f_matrix<var<2U> > second_order_func;
  
  start = clock();
  for (int n = 0; n < n_calls; ++n)
    hessian(second_order_func, x, f, grad, H);
  end = clock();
  
  deltaT = (double)(end - start) / CLOCKS_PER_SEC;
  
  std::cout << n_calls << " hessians took " << deltaT << " seconds" << std::endl;
  
  // Third-order timing
  f_matrix<var<3U> > third_order_func;
  
  start = clock();
  for (int n = 0; n < n_calls; ++n)
    grad_trace_matrix_times_hessian(third_order_func, x, M, f, grad, H, grad_m_times_h);
  end = clock();
  
  deltaT = (double)(end - start) / CLOCKS_PER_SEC;
  
  std::cout << n_calls << " grad-matrix-times-hessians took " << deltaT << " seconds" << std::endl;
  
}
*/

int main(int argc, const char * argv[]) {
  (void)argc; (void)argv;
  
  /*
  Eigen::VectorXd x = Eigen::VectorXd::Ones(6);
  x(0) = 0;
  
  Eigen::VectorXd v = Eigen::VectorXd::Ones(6);
  Eigen::MatrixXd M = Eigen::MatrixXd::Identity(6, 6);
  
  double f;
  Eigen::VectorXd grad(x.size());
  Eigen::MatrixXd H(x.size(), x.size());
  Eigen::VectorXd grad_m_times_h(x.size());
  
  debug_func<debug_var1, 5> first_order_funnel;
  try {
    gradient(first_order_funnel, x, f, grad);
  } catch(nomad_error& e) {
    std::cout << e.what() << std::endl;
  }
  
  std::cout << f << std::endl;
  std::cout << grad.transpose() << std::endl;
  */
  
  //validate_exceptions();
  
  //validate_funnel();
  time_funnel();
  
  //validate_matrix();
  //time_matrix();
  
  //validate_dot();
  //time_dot();
  
  return 0;
}



