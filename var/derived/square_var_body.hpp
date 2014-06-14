#ifndef nomad__var__derived__square_var_body_hpp
#define nomad__var__derived__square_var_body_hpp

#include <var/var_body.hpp>

namespace nomad {

  template<short autodiff_order>
  class square_var_body: public var_base {
  public:
    
    // Experiment with moving the checks back into the constructor proper
    static inline void* operator new(size_t /* ignore */) {
      if (unlikely(next_body_idx_ + 1 > max_body_idx)) expand_var_bodies<autodiff_order>();
      // no partials
      if (unlikely(next_inputs_idx_ + next_inputs_delta > max_inputs_idx)) expand_inputs();
      return var_bodies_ + next_body_idx_;
    }
    
    static inline void operator delete(void* /* ignore */) {};
    
    square_var_body(): var_base(1) {};
    
    inline unsigned int n_first_partials() { return 0; }
    inline unsigned int n_second_partials() { return 0; }
    inline unsigned int n_third_partials() { return 0; }
    inline static unsigned int n_partials(unsigned int n_inputs) { return 0; }
    
    inline void first_order_forward_adj() {
      if (autodiff_order >= 1) {
        first_grad() = 2 * first_grad(input()) * first_val(input());
      }
    }
    
    inline void first_order_reverse_adj() {
      if (autodiff_order >= 1) {
        first_grad(input()) += 2 * first_grad() * first_val(input());
      }
    }
    
    virtual void second_order_forward_val() {
      if (autodiff_order >= 2) {
        second_grad() = 0;
        second_val() = 2 * second_val(input()) * first_val(input());
      }
    }
    
    virtual void second_order_reverse_adj() {
      if (autodiff_order >= 2) {
        second_grad(input()) +=   2 * second_grad() * first_val(input())
                                + 2 * first_grad() * second_val(input());
      }
    }
    
    virtual void third_order_forward_val() {
      if (autodiff_order >= 3) {
        third_grad() = 0;
        fourth_grad() = 0;
        third_val()  =   2 * third_val(input()) * first_val(input());
        fourth_val() =   2 * fourth_val(input()) * first_val(input())
                       + 2 * third_val(input())  * second_val(input());
      }
      
    } // third_order_forward_val
    
    virtual void third_order_reverse_adj() {
      
      if (autodiff_order >= 3) {
        
        const double g1 = first_grad();
        const double g2 = second_grad();
        const double g3 = third_grad();
        const double g4 = fourth_grad();
        
        third_grad(input()) +=   2 * third_grad() * first_val(input())
        + 2 * first_grad() * third_val(input());
        
        fourth_grad(input()) +=   2 * fourth_grad() * first_val(input())
                                + 2 * third_grad()  * second_val(input())
                                + 2 * second_grad() * third_val(input())
                                + 2 * first_grad()  * fourth_val(input());
        
      }
      
    } // third_order_reverse_adj
    
  };
  
}

#endif
