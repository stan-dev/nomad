#ifndef nomad__matrix__functions__dot_hpp
#define nomad__matrix__functions__dot_hpp

#include <Eigen/Core>

#include <var/var.hpp>
#include <var/derived/dot_var_body.hpp>

namespace nomad {
  
  template<typename DerivedA, typename DerivedB>
  inline typename
  std::enable_if<
    is_var<typename Eigen::MatrixBase<DerivedA>::Scalar>::value &&
    is_var<typename Eigen::MatrixBase<DerivedB>::Scalar>::value &&
    std::is_same<typename Eigen::MatrixBase<DerivedA>::Scalar,
                 typename Eigen::MatrixBase<DerivedB>::Scalar>::value,
    typename Eigen::MatrixBase<DerivedA>::Scalar >::type
  dot(const Eigen::MatrixBase<DerivedA>& v1,
      const Eigen::MatrixBase<DerivedB>& v2) {
    
    const short autodiff_order = Eigen::MatrixBase<DerivedA>::Scalar::order();

    eigen_idx_t N = v1.size();
    const nomad_idx_t n_inputs = static_cast<nomad_idx_t>(2 * N);
    
    next_inputs_delta = n_inputs;
    // next_partials_delta not used by dot_var_body
    
    new dot_var_body<autodiff_order>(n_inputs);
    
    double sum = 0;
    
    for (eigen_idx_t n = 0; n < N; ++n)
      sum += v1(n).first_val() * v2(n).first_val();

    push_dual_numbers<autodiff_order>(sum);
    
    for (eigen_idx_t n = 0; n < N; ++n)
      push_inputs(v1(n).dual_numbers());
      
    for (eigen_idx_t n = 0; n < N; ++n)
      push_inputs(v2(n).dual_numbers());
    
    return var<autodiff_order>(next_body_idx_ - 1);
    
  }
  
}

#endif
