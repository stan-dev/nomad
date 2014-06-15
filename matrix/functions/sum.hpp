#ifndef nomad__matrix__functions__sum_hpp
#define nomad__matrix__functions__sum_hpp

#include <Eigen/Core>

#include <var/var.hpp>
#include <var/derived/sum_var_body.hpp>

namespace nomad {
  
  template<typename Derived>
  inline typename std::enable_if<
    is_var<typename Eigen::MatrixBase<Derived>::Scalar>::value,
    typename Eigen::MatrixBase<Derived>::Scalar >::type
  sum(const Eigen::MatrixBase<Derived>& input) {
    
    const short autodiff_order = Eigen::MatrixBase<Derived>::Scalar::order();
    const short partials_order = 1;
    const unsigned int n_inputs = input.size();
    
    next_inputs_delta = n_inputs;
    // next_partials_delta not used by sum_var_body
    
    new sum_var_body<autodiff_order>(n_inputs);
    
    double sum = 0;
    
    for (int n = 0; n < n_inputs; ++n)
      sum += input(n).first_val();
    push_dual_numbers<autodiff_order>(sum);
    
    for (int n = 0; n < n_inputs; ++n)
      push_inputs(input(n).dual_numbers());
    
    return var<autodiff_order>(next_body_idx_ - 1);
    
  }

}

#endif
