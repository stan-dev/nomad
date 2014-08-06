#ifndef nomad__src__var__var_hpp
#define nomad__src__var__var_hpp

#include <type_traits>

#include <src/var/var_body.hpp>

namespace nomad {
  
  template <short autodiff_order, bool strict_smoothness>
  class var {
  private:

    nomad_idx_t body_idx_;
    
  public:
    
    var() {}
    var(const var& v) : body_idx_(v.body_idx_) {}
    
    explicit var(nomad_idx_t body_idx) : body_idx_(body_idx) {}
    
    var(double val) {
      // next_partials_delta not used by var_body<autodiff_order, 0>
      // next_inputs_delta not used by var_body<autodiff_order, 0>
      new var_body<autodiff_order, 0>();
      push_dual_numbers<autodiff_order>(val);
      body_idx_ = next_body_idx_ - 1;
    }
    
    var& operator=(const var& v) {
      body_idx_ = v.body_idx_;
      return *this;
    }

    var& operator=(double val) {
      // next_partials_delta not used by var_body<autodiff_order, 0>
      // next_inputs_delta not used by var_body<autodiff_order, 0>
      new var_body<autodiff_order, 0>();
      push_dual_numbers<autodiff_order>(val);
      body_idx_ = next_body_idx_ - 1;
      return *this;
    }
    
    nomad_idx_t dual_numbers() const { return var_bodies_[body_idx_].dual_numbers(); }
    
    nomad_idx_t body() const { return body_idx_; }
    void set_body(nomad_idx_t body_idx) { body_idx_ = body_idx; }
    
    constexpr static short order() { return autodiff_order; }
    constexpr static bool strict() { return strict_smoothness; }
    
    double& first_val()   const { return var_bodies_[body_idx_].first_val(); }
    double& first_grad()  const { return var_bodies_[body_idx_].first_grad(); }
    double& second_val()  const { return var_bodies_[body_idx_].second_val(); }
    double& second_grad() const { return var_bodies_[body_idx_].second_grad(); }
    double& third_val()   const { return var_bodies_[body_idx_].third_val(); }
    double& third_grad()  const { return var_bodies_[body_idx_].third_grad(); }
    double& fourth_val()  const { return var_bodies_[body_idx_].fourth_val(); }
    double& fourth_grad() const { return var_bodies_[body_idx_].fourth_grad(); }
    
  };

  template <typename T>
  struct is_var : public std::false_type { };
  
  template <short autodiff_order, bool strict_smoothness>
  struct is_var< var<autodiff_order, strict_smoothness> > : public std::true_type { };
  
  typedef var<1, true> var1;
  typedef var<2, true> var2;
  typedef var<3, true> var3;
  
  typedef var<1, false> wild_var1;
  typedef var<2, false> wild_var2;
  typedef var<3, false> wild_var3;

}

#endif
