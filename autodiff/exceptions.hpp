#ifndef nomad__autodiff__exceptions_hpp
#define nomad__autodiff__exceptions_hpp

#include <string>
#include <exception>

namespace nomad {

  class partial_fail_ex: public std::runtime_error {
  public:
    partial_fail_ex(std::string error):
      std::runtime_error(error) {}
  };

  class autodiff_fail_ex: public std::runtime_error {
  public:
    autodiff_fail_ex(std::string error):
      std::runtime_error(error) {}
  };
  
}

#endif
