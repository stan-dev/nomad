#ifndef nomad__src__var__derived__unary_var_node_hpp
#define nomad__src__var__derived__unary_var_node_hpp

#include <src/var/var_node.hpp>

namespace nomad {
  
  template<short AutodiffOrder, short PartialsOrder>
  class unary_var_node: public var_node_base {
  public:
    
    static inline void* operator new(size_t /* ignore */) {
      if (unlikely(next_node_idx_ + 1 > max_node_idx)) expand_var_nodes<AutodiffOrder>();
      if (unlikely(next_partials_idx_ + next_partials_delta > max_partials_idx)) expand_partials();
      if (unlikely(next_inputs_idx_ + next_inputs_delta > max_inputs_idx)) expand_inputs();
      return var_nodes_ + next_node_idx_;
    }
    
    static inline void operator delete(void* /* ignore */) {}
    
    unary_var_node(): var_node_base(1) {}
    
    constexpr static bool dynamic_inputs() { return false; }

    inline nomad_idx_t n_first_partials() {
      return AutodiffOrder >= 1 && PartialsOrder >= 1 ? 1 : 0;
    }
    
    inline nomad_idx_t n_second_partials() {
      return AutodiffOrder >= 2 && PartialsOrder >= 2 ? 1 : 0;
    }
    
    inline nomad_idx_t n_third_partials() {
      return AutodiffOrder >= 3 && PartialsOrder >= 3 ? 1 : 0;
    }

    inline static nomad_idx_t n_partials() {
      if (AutodiffOrder >= 1 && PartialsOrder >= 1) return 1;
      if (AutodiffOrder >= 2 && PartialsOrder >= 2) return 2;
      if (AutodiffOrder >= 3 && PartialsOrder >= 3) return 3;
      return 0;
    }
    
    inline static nomad_idx_t n_partials(nomad_idx_t n_inputs) {
      (void)n_inputs;
      if (AutodiffOrder >= 1 && PartialsOrder >= 1) return 1;
      if (AutodiffOrder >= 2 && PartialsOrder >= 2) return 2;
      if (AutodiffOrder >= 3 && PartialsOrder >= 3) return 3;
      return 0;
    }
    
    inline void first_order_forward_adj() {
      first_grad() = 0;
      if (PartialsOrder >= 1)
        first_grad() += first_grad(input()) * first_partials(0);
    }
    
    inline void first_order_reverse_adj() {
      if (PartialsOrder >= 1)
        first_grad(input()) += first_grad() * first_partials(0);
    }
    
    void second_order_forward_val() {
      
      if (AutodiffOrder >= 2) {
        
        second_val() = 0;
        second_grad() = 0;
        
        if (PartialsOrder >= 1)
          second_val() += second_val(input()) * first_partials(0);
        
      }
      
    }
    
    void second_order_reverse_adj() {
      
      if (AutodiffOrder >= 2) {
        
        if (PartialsOrder >= 1)
            second_grad(input()) += second_grad() * first_partials(0);
        
        if (PartialsOrder >= 2)
          second_grad(input()) += first_grad() * second_val(input()) * first_partials(1);
        
      }
      
    }
    
    void third_order_forward_val() {
      
      if (AutodiffOrder >= 3) {
        
        third_val() = 0;
        fourth_val() = 0;

        third_grad() = 0;
        fourth_grad() = 0;
        
        if (PartialsOrder >= 1) {
          third_val() += third_val(input()) * first_partials(0);
          fourth_val() += fourth_val(input()) * first_partials(0);
        }
        
        if (PartialsOrder >= 2)
          fourth_val() += second_val(input()) * third_val(input()) * first_partials(1);
        
      }
      
    } // third_order_forward_val
    
    void third_order_reverse_adj() {
      
      if (AutodiffOrder >= 3) {

        const double g1 = first_grad();
        const double g2 = second_grad();
        const double g3 = third_grad();
        const double g4 = fourth_grad();
        
        if (PartialsOrder >= 1) {
          third_grad(input())  += g3 * first_partials(0);
          fourth_grad(input()) += g4 * first_partials(0);
        }
        
        if (PartialsOrder >= 2) {
          third_grad(input())  += g1 * third_val(input()) * first_partials(1);
          fourth_grad(input()) += (   g1 * fourth_val(input())
                                    + g2 * third_val(input())
                                    + g3 * second_val(input()) ) * first_partials(1);
        }
        
        if(PartialsOrder >= 3)
          fourth_grad(input()) += g1 * second_val(input())
                                  * third_val(input()) * first_partials(2);
        
      }
      
    } // third_order_reverse_adj
    
  };
  
}

#endif
