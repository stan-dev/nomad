#ifndef nomad__src__validation__sizes_hpp
#define nomad__src__validation__sizes_hpp

#include <src/validation/exceptions.hpp>

namespace nomad {

  /*
  template <typename Head, typename… Tail>
  bool consistent_sizes(const std::string& function,
                        Head& head, Tail…& tail) {
    size_t max = max_size(head, tail);
    check_consistent_sizes(function, max_size, head, tail);
  }
  
  template <typename Head, typename… Tail>
  bool consistent_sizes(const std::string& function,
                        size_t max_size,
                        Head& head, Tail…& tail) {
    check_consistent_sizes(function, max_size, tail);
  }
  
  template <typename T>
  inline bool check_consistent_size(const std::string& function,
                                    size_t max_size, T wrapper) {
    size_t x_size = stan::size_of(wrapper::x);
    if (x_size == 1 || x_size == max_size)
      
      std::stringstream msg;
    msg << ", expecting dimension of either 1 or "
    << "max_size=" << max_size
    << "; a vectorized function was called with arguments of different "
    << "scalar, array, vector, or matrix types, and they were not "
    << "consistently sized;  all arguments must be scalars or "
    << "multidimensional values of the same shape.";
    
    dom_err(function, wrapper::name, x_size, "dimension=", msg.str());
    
    return false;
  }
  */
}

#endif
