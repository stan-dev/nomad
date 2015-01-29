#ifndef nomad__src__autodiff__exceptions_hpp
#define nomad__src__autodiff__exceptions_hpp

#include <string>
#include <exception>

namespace nomad {

  class nomad_error: public std::runtime_error {
  public:
    nomad_error(std::string error = "Generic Nomad Error"): std::runtime_error(error) {}
  private:
    virtual void anchor_vtable();
  };
  void nomad_error::anchor_vtable() {}
  
  class nomad_domain_error: public nomad_error {
  public:
    nomad_domain_error(double input, std::string domain, std::string name):
    nomad_error("Nomad stack construction terminated because the input "
                + std::to_string(input) + " violated the " + domain +
                " of " + name) {}
    
    nomad_domain_error(std::string domain, std::string name):
    nomad_error("Nomad stack construction terminated because the input(s) violated the "
                + domain + " of " + name) {}
  private:
    virtual void anchor_vtable();
  };
  void nomad_domain_error::anchor_vtable() {}
  
  class nomad_input_error: public nomad_error {
  public:
    nomad_input_error(std::string name):
    nomad_error("Nomad stack construction terminated because "
                + name + " encountered a NaN input") {}
  private:
    virtual void anchor_vtable();
  };
  void nomad_input_error::anchor_vtable() {}
  
  class nomad_output_value_error: public nomad_error {
  public:
    nomad_output_value_error(std::string name):
    nomad_error("Nomad stack construction terminated because "
                + name + " generated a NaN value") {}
  private:
    virtual void anchor_vtable();
  };
  void nomad_output_value_error::anchor_vtable() {}
  
  class nomad_output_partial_error: public nomad_error {
  public:
    nomad_output_partial_error(std::string name):
    nomad_error("Nomad stack construction terminated because "
                + name + " generated a NaN partial") {}
  private:
    virtual void anchor_vtable();
  };
  void nomad_output_partial_error::anchor_vtable() {}
  
}

#endif
