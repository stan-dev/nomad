#ifndef nomad__src__matrix__functions__sum_hpp
#define nomad__src__matrix__functions__sum_hpp

#include <Eigen/Core>

#include <src/var/var.hpp>
#include <src/var/derived/multi_sum_var_node.hpp>

namespace nomad {
  
  template<typename Derived>
  inline typename std::enable_if<
    is_var<typename Eigen::MatrixBase<Derived>::Scalar>::value,
           typename Eigen::MatrixBase<Derived>::Scalar >::type
  sum(const Eigen::MatrixBase<Derived>& input) {
    
    const short AutodiffOrder = Eigen::MatrixBase<Derived>::Scalar::order();
    const bool StrictSmoothness = Eigen::MatrixBase<Derived>::Scalar::strict();
    const nomad_idx_t n_inputs = static_cast<nomad_idx_t>(input.size());
    
    next_inputs_delta = n_inputs;
    // next_partials_delta not used by multi_sum_var_node
    
    new multi_sum_var_node<AutodiffOrder>(n_inputs);
    
    double sum = 0;
    
    for (eigen_idx_t n = 0; n < n_inputs; ++n)
      sum += input(n).first_val();
    push_dual_numbers<AutodiffOrder>(sum);
    
    for (eigen_idx_t n = 0; n < n_inputs; ++n)
      push_inputs(input(n).dual_numbers());
    
    return var<AutodiffOrder, StrictSmoothness>(next_body_idx_ - 1);
    
  }

}

#endif
