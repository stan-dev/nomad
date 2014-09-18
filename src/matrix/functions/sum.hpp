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
    
    const short autodiff_order = Eigen::MatrixBase<Derived>::Scalar::order();
    const bool strict_smoothness = Eigen::MatrixBase<Derived>::Scalar::strict();
    const bool validate_io = Eigen::MatrixBase<Derived>::Scalar::validate();
    
    const nomad_idx_t n_inputs = static_cast<nomad_idx_t>(input.size());
    
    create_node<multi_sum_var_node<autodiff_order>>(n_inputs);
    
    double sum = 0;
    
    for (eigen_idx_t n = 0; n < n_inputs; ++n)
      sum += input(n).first_val();
    push_dual_numbers<autodiff_order>(sum);
    
    for (eigen_idx_t n = 0; n < n_inputs; ++n)
      push_inputs(input(n).dual_numbers());
    
    return var<autodiff_order, strict_smoothness, validate_io>(next_node_idx_ - 1);
    
  }

}

#endif
