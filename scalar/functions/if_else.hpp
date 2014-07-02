#ifndef nomad__scalar__functions__if_else_hpp
#define nomad__scalar__functions__if_else_hpp

#include <var/var.hpp>

namespace nomad {
  
  template <short autodiff_order>
  inline var<autodiff_order> if_else(bool c,
                                     const var<autodiff_order>& v_true,
                                     const var<autodiff_order>& v_false) {
    return c ? v_true : v_false;
  }
  
  template <short autodiff_order>
  inline var<autodiff_order> if_else(bool c,
                                     double x_true,
                                     const var<autodiff_order>& v_false) {
    return c ? x_true : v_false;
  }
  
  template <short autodiff_order>
  inline var<autodiff_order> if_else(bool c,
                                     const var<autodiff_order>& v_true,
                                     double x_false) {
    return c ? v_true : x_false;
  }

}

#endif
