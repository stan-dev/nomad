#ifndef nomad__src__scalar__functions__smooth_functions__Phi_approx_hpp
#define nomad__src__scalar__functions__smooth_functions__Phi_approx_hpp

#include <src/var/var.hpp>
#include <src/scalar/functions/inv_logit.hpp>
#include <src/scalar/functions/pow.hpp>

namespace nomad {
  
  // Approximation of the unit normal CDF for variables.
  // http://www.jiem.org/index.php/jiem/article/download/60/27
  
  inline double Phi_approx(double x) {
    return inv_logit(0.07056 * pow(x, 3.0) + 1.5976 * x);
  }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    Phi_approx(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& input) {
      return inv_logit(0.07056 * pow(input, 3.0) + 1.5976 * input);
    
  }

}

#endif
