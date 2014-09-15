#ifndef nomad__src__autodiff__exceptions_hpp
#define nomad__src__autodiff__exceptions_hpp

#include <string>
#include <exception>

namespace nomad {

  class nomad_error: public std::runtime_error {
  public:
    nomad_error(std::string error = "Generic Nomad Error"): std::runtime_error(error) {}
  };
  
  class nomad_domain_error: public nomad_error {
  public:
    nomad_domain_error(std::string name):
    nomad_error("Nomad stack construction terminated because of a domain "
                + std::string("error in the inputs to the function ") + name) {}
  };

  class nomad_input_error: public nomad_error {
  public:
    nomad_input_error(std::string name):
    nomad_error("Nomad stack construction terminated because the function "
                + name + " encountered a NaN input") {}
  };
  
  class nomad_output_error: public nomad_error {
  public:
    nomad_output_error(std::string name):
    nomad_error("Nomad stack construction terminated because the function "
                + name + " generated a NaN value or partial") {}
  };
  
  class nomad_output_value_error: public nomad_error {
  public:
    nomad_output_value_error(std::string name):
    nomad_error("Nomad stack construction terminated because the function "
                + name + " generated a NaN value") {}
  };
  
  class nomad_output_partial_error: public nomad_error {
  public:
    nomad_output_partial_error(std::string name):
    nomad_error("Nomad stack construction terminated because the function "
                + name + " generated a NaN partial") {}
  };
  
}

#endif