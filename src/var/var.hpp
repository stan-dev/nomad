#ifndef nomad__src__var__var_hpp
#define nomad__src__var__var_hpp

#include <type_traits>

#include <src/var/var_node.hpp>
#include <src/autodiff/validation.hpp>

namespace nomad {
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  class var {
  private:

    nomad_idx_t node_idx_;
    
  public:
    
    var() {}
    var(const var& v) : node_idx_(v.node_idx_) {}
    
    explicit var(nomad_idx_t node_idx) : node_idx_(node_idx) {}
    
    var(double val) {
      if (ValidateIO) validate_input(val, "var constructor");
      create_node<var_node<AutodiffOrder, 0>>(0);
      push_dual_numbers<AutodiffOrder, ValidateIO>(val);
      node_idx_ = next_node_idx_ - 1;
    }
    
    var& operator=(const var& v) {
      node_idx_ = v.node_idx_;
      return *this;
    }

    var& operator=(double val) {
      if (ValidateIO) validate_input(val, "var operator=");
      create_node<var_node<AutodiffOrder, 0>>(0);
      push_dual_numbers<AutodiffOrder, ValidateIO>(val);
      node_idx_ = next_node_idx_ - 1;
      return *this;
    }
    
    nomad_idx_t dual_numbers() const { return var_nodes_[node_idx_].dual_numbers(); }
    
    nomad_idx_t node() const { return node_idx_; }
    void set_node(nomad_idx_t node_idx) { node_idx_ = node_idx; }
    
    constexpr static short order() { return AutodiffOrder; }
    constexpr static bool strict() { return StrictSmoothness; }
    constexpr static bool validate() { return ValidateIO; }
    
    double& first_val()   const { return var_nodes_[node_idx_].first_val(); }
    double& first_grad()  const { return var_nodes_[node_idx_].first_grad(); }
    double& second_val()  const { return var_nodes_[node_idx_].second_val(); }
    double& second_grad() const { return var_nodes_[node_idx_].second_grad(); }
    double& third_val()   const { return var_nodes_[node_idx_].third_val(); }
    double& third_grad()  const { return var_nodes_[node_idx_].third_grad(); }
    double& fourth_val()  const { return var_nodes_[node_idx_].fourth_val(); }
    double& fourth_grad() const { return var_nodes_[node_idx_].fourth_grad(); }
    
  };

  template <typename T>
  struct is_var : public std::false_type { };
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  struct is_var< var<AutodiffOrder, StrictSmoothness, ValidateIO> > : public std::true_type { };
  
  typedef var<1, true, false> var1;
  typedef var<2, true, false> var2;
  typedef var<3, true, false> var3;

  typedef var<1, true, true> debug_var1;
  typedef var<2, true, true> debug_var2;
  typedef var<3, true, true> debug_var3;
  
  typedef var<1, false, false> wild_var1;
  typedef var<2, false, false> wild_var2;
  typedef var<3, false, false> wild_var3;

}

#endif
