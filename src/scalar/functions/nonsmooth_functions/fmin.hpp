#ifndef nomad__src__scalar__functions__nonsmooth_functions__fmin_hpp
#define nomad__src__scalar__functions__nonsmooth_functions__fmin_hpp

#include <math.h>
#include <type_traits>
#include <src/var/var.hpp>
#include <src/autodiff/validation.hpp>

namespace nomad {
  
  inline double fmin(double x, double y) { return std::fmin(x, y); }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline typename std::enable_if<!StrictSmoothness, var<AutodiffOrder, StrictSmoothness, ValidateIO> >::type
    fmin(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v1,
         const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v2) {
    if (ValidateIO) {
      validate_input(v1.first_val(), "fmin");
      validate_input(v2.first_val(), "fmin");
    }
    return v1.first_val() < v2.first_val() ? v1 : v2;
  }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline typename std::enable_if<!StrictSmoothness, var<AutodiffOrder, StrictSmoothness, ValidateIO> >::type
    fmin(double x,
         const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v2) {
    if (ValidateIO) {
      validate_input(x, "fmin");
      validate_input(v2.first_val(), "fmin");
    }
    return x < v2.first_val() ? var<AutodiffOrder, StrictSmoothness, ValidateIO>(x) : v2;
  }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline typename std::enable_if<!StrictSmoothness, var<AutodiffOrder, StrictSmoothness, ValidateIO> >::type
    fmin(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v1,
         double y) {
    if (ValidateIO) {
      validate_input(v1.first_val(), "fmin");
      validate_input(y, "fmin");
    }
    return v1.first_val() < y ? v1 : var<AutodiffOrder, StrictSmoothness, ValidateIO>(y);
  }

}

#endif
