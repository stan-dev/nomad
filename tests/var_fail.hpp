#ifndef nomad__tests__var_fail_hpp
#define nomad__tests__var_fail_hpp

#include <var/var.hpp>
#include <autodiff/exceptions.hpp>

namespace nomad {

  template<typename T>
  inline void first_fail() {
    if (T::order() >= 1)
      throw partial_fail_ex("No partials have "
                            "been implemented for first_fail");
  }

  template<typename T>
  inline void second_fail() {
    if (T::order() >= 2)
      throw partial_fail_ex("Only first-order partials have "
                            "been implemented for second_fail");
  }

  template<typename T>
  inline void third_fail() {
    if (T::order() >= 3)
      throw partial_fail_ex("Only second-order partials have "
                            "been implemented for third_fail");
  }

}
  
#endif
