#ifndef nomad__var__derived__unary_minus_var_body_hpp
#define nomad__var__derived__unary_minus_var_body_hpp

#include <var/var_body.hpp>

namespace nomad {
  
  template<short autodiff_order>
  class unary_minus_var_body: public var_base {
  public:
    
    static inline void* operator new(size_t /* ignore */) {
      if (unlikely(next_body_idx_ + 1 > max_body_idx)) expand_var_bodies<autodiff_order>();
      // no partials
      if (unlikely(next_inputs_idx_ + next_inputs_delta > max_inputs_idx)) expand_inputs();
      return var_bodies_ + next_body_idx_;
    }
    
    static inline void operator delete(void* /* ignore */) {}
    
    unary_minus_var_body(): var_base(1) {}

    inline nomad_idx_t n_first_partials() { return 0; }
    inline nomad_idx_t n_second_partials() { return 0; }
    inline nomad_idx_t n_third_partials() { return 0; }
    inline static nomad_idx_t n_partials() { return 0; }
    inline static nomad_idx_t n_partials(unsigned int n_inputs) { return 0; }
    
    inline void first_order_forward_adj() {
      if (autodiff_order >= 1)
        first_grad() = -first_grad(input());
    }
    
    inline void first_order_reverse_adj() {
      if (autodiff_order >= 1) first_grad(input()) += -first_grad();
    }
    
    inline void second_order_forward_val() {
      if (autodiff_order >= 2) {
        second_val() = -second_val(input());
        second_grad() = 0;
      }
    }
    
    inline void second_order_reverse_adj() {
      if (autodiff_order >= 2) second_grad(input()) += -second_grad();
    }
    
    inline void third_order_forward_val() {
      if (autodiff_order >= 3) {
        third_val() = -third_val(input());
        fourth_val() = -fourth_val(input());
        third_grad() = 0;
        fourth_grad() = 0;
      }
    }
    
    inline void third_order_reverse_adj() {
      if (autodiff_order >= 3) {
        third_grad(input()) += -third_grad();
        fourth_grad(input()) += -fourth_grad();
      }
    }
    
  };
  
}

#endif
