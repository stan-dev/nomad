#ifndef nomad__src__var__derived__binary_var_body_hpp
#define nomad__src__var__derived__binary_var_body_hpp

#include <src/var/var_body.hpp>

namespace nomad {
  
  template<short autodiff_order, short partials_order>
  class binary_var_body: public var_base {
  public:
    
    static inline void* operator new(size_t /* ignore */) {
      if (unlikely(next_body_idx_ + 1 > max_body_idx)) expand_var_bodies<autodiff_order>();
      if (unlikely(next_partials_idx_ + next_partials_delta > max_partials_idx)) expand_partials();
      if (unlikely(next_inputs_idx_ + next_inputs_delta > max_inputs_idx)) expand_inputs();
      return var_bodies_ + next_body_idx_;
    }
    
    static inline void operator delete(void* /* ignore */) {}
    
    binary_var_body(): var_base(2) {}

    constexpr static bool dynamic_inputs() { return false; }
    
    inline nomad_idx_t n_first_partials() {
      return autodiff_order >= 1 && partials_order >= 1 ? 2 : 0;
    }
    
    inline nomad_idx_t n_second_partials() {
      return autodiff_order >= 2 && partials_order >= 2 ? 3 : 0;
    }
    
    inline nomad_idx_t n_third_partials() {
      return autodiff_order >= 3 && partials_order >= 3 ? 4 : 0;
    }

    inline static nomad_idx_t n_partials() {
      if (autodiff_order >= 1 && partials_order >= 1) return 2;
      if (autodiff_order >= 2 && partials_order >= 2) return 5;
      if (autodiff_order >= 3 && partials_order >= 3) return 9;
      return 0;
    }
    
    inline static nomad_idx_t n_partials(nomad_idx_t n_inputs) {
      if (autodiff_order >= 1 && partials_order >= 1) return 2;
      if (autodiff_order >= 2 && partials_order >= 2) return 5;
      if (autodiff_order >= 3 && partials_order >= 3) return 9;
      return 0;
    }
    
    inline void first_order_forward_adj() {
      first_grad() = 0;
      if (partials_order >= 1)
        first_grad() +=   first_grad(input())  * first_partials(0)
                        + first_grad(input(1)) * first_partials(1);
    }
    
    inline void first_order_reverse_adj() {
      if (partials_order >= 1) {
        const double g = first_grad();
        first_grad(input()) += g * first_partials(0);
        first_grad(input(1)) += g * first_partials(1);
      }
    }
    
    void second_order_forward_val() {
      
      if (autodiff_order >= 2) {
        second_val() = 0;
        second_grad() = 0;
      
        if (partials_order >= 1)
          second_val() +=   second_val(input())  * first_partials(0)
                          + second_val(input(1)) * first_partials(1);
      }
      
    }
    
    void second_order_reverse_adj() {
      
      if (autodiff_order >= 2) {
        
        if (partials_order >= 1) {
          second_grad(input())  += second_grad() * first_partials(0);
          second_grad(input(1)) += second_grad() * first_partials(1);
        }
        
        if (partials_order >= 2) {
          second_grad(input())  += first_grad() * (  second_val(input())  * first_partials(2)
                                                   + second_val(input(1)) * first_partials(3));
          second_grad(input(1)) += first_grad() * (  second_val(input())  * first_partials(3)
                                                   + second_val(input(1)) * first_partials(4));
        }
        
      }
      
    }
    
    void third_order_forward_val() {
      
      if (autodiff_order >= 3) {
        third_val() = 0;
        fourth_val() = 0;
        third_grad() = 0;
        fourth_grad() = 0;
        
        if (partials_order >= 1) {
          third_val()  +=   third_val(input())  * first_partials(0)
                          + third_val(input(1)) * first_partials(1);
          
          fourth_val() +=   fourth_val(input())  * first_partials(0)
                          + fourth_val(input(1)) * first_partials(1);
        }
        
        if (partials_order >= 2) {
          fourth_val() +=   second_val(input()) * third_val(input())   * first_partials(2)
                          + second_val(input()) * third_val(input(1))  * first_partials(3)
                          + second_val(input(1)) * third_val(input())  * first_partials(3)
                          + second_val(input(1)) * third_val(input(1)) * first_partials(4);
        }
        
      }
      
    } // third_order_forward_val
    
    void third_order_reverse_adj() {
      
      if (autodiff_order >= 3) {

        const double g1 = first_grad();
        const double g2 = second_grad();
        const double g3 = third_grad();
        const double g4 = fourth_grad();
        
        if (partials_order >= 1) {
          third_grad(input())   += g3 * first_partials(0);
          third_grad(input(1))  += g3 * first_partials(1);
          
          fourth_grad(input())  += g4 * first_partials(0);
          fourth_grad(input(1)) += g4 * first_partials(1);
        }
        
        if (partials_order >= 2) {
          third_grad(input())  += g1 * (  third_val(input())  * first_partials(2)
                                        + third_val(input(1)) * first_partials(3));
          
          third_grad(input(1)) += g1 * (  third_val(input(1)) * first_partials(3)
                                        + third_val(input())  * first_partials(4));
          
          fourth_grad(input())  +=  (  g1 * fourth_val(input())
                                     + g2 * third_val(input())
                                     + g3 * second_val(input())) * first_partials(2)
                                  + (  g1 * fourth_val(input(1))
                                     + g2 * third_val(input(1))
                                     + g3 * second_val(input(1))) * first_partials(3);
          
          fourth_grad(input(1)) +=  (  g1 * fourth_val(input())
                                     + g2 * third_val(input())
                                     + g3 * second_val(input())) * first_partials(3)
                                  + (  g1 * fourth_val(input(1))
                                     + g2 * third_val(input(1))
                                     + g3 * second_val(input(1))) * first_partials(4);
        }
        
        if(partials_order >= 3) {
          fourth_grad(input())  +=   g1 * second_val(input(0)) * third_val(input(0)) * first_partials(5)
                                   + g1 * second_val(input(0)) * third_val(input(1)) * first_partials(6)
                                   + g1 * second_val(input(1)) * third_val(input(0)) * first_partials(6)
                                   + g1 * second_val(input(1)) * third_val(input(1)) * first_partials(7);
          fourth_grad(input(1)) +=   g1 * second_val(input(0)) * third_val(input(0)) * first_partials(6)
                                   + g1 * second_val(input(0)) * third_val(input(1)) * first_partials(7)
                                   + g1 * second_val(input(1)) * third_val(input(0)) * first_partials(7)
                                   + g1 * second_val(input(1)) * third_val(input(1)) * first_partials(8);
        }
      
      }
      
    } // third_order_reverse_adj
    
  };
  
}

#endif
