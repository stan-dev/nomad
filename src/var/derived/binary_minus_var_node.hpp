#ifndef nomad__src__var__derived__binary_minus_var_node_hpp
#define nomad__src__var__derived__binary_minus_var_node_hpp

#include <src/var/var_node.hpp>

namespace nomad {

  template<short AutodiffOrder>
  class binary_minus_var_node: public var_node_base {
  public:
    
    static inline void* operator new(size_t /* ignore */) {
      if (unlikely(next_node_idx_ + 1 > max_node_idx)) expand_var_nodes<AutodiffOrder>();
      // no partials
      if (unlikely(next_inputs_idx_ + next_inputs_delta > max_inputs_idx)) expand_inputs();
      return var_nodes_ + next_node_idx_;
    }
    
    static inline void operator delete(void* /* ignore */) {}
    
    binary_minus_var_node(): var_node_base(2) {}
 
    constexpr static bool dynamic_inputs() { return false; }
    
    inline nomad_idx_t n_first_partials() { return 0; }
    inline nomad_idx_t n_second_partials() { return 0; }
    inline nomad_idx_t n_third_partials() { return 0; }
    inline static nomad_idx_t n_partials() { return 0; }
    inline static nomad_idx_t n_partials(unsigned int n_inputs) { (void)n_inputs; return 0; }
    
    inline void first_order_forward_adj() {
      if (AutodiffOrder >= 1)
        first_grad() = first_grad(input()) - first_grad(input(1));
    }
    
    inline void first_order_reverse_adj() {
      if (AutodiffOrder >= 1) {
        const double g1 = first_grad();
        first_grad(input())  += g1;
        first_grad(input(1)) -= g1;
      }
    }
    
    void second_order_forward_val() {
      if (AutodiffOrder >= 2) {
        second_val() = second_val(input()) - second_val(input(1));
        second_grad() = 0;
      }
    }
    
    void second_order_reverse_adj() {
      if (AutodiffOrder >= 2) {
        const double g2 = second_grad();
        second_grad(input())  += g2;
        second_grad(input(1)) -= g2;
      }
    }
    
    void third_order_forward_val() {
      
      if (AutodiffOrder >= 3) {
        third_val() = third_val(input()) - third_val(input(1));
        fourth_val() = fourth_val(input()) - fourth_val(input(1));
        third_grad() = 0;
        fourth_grad() = 0;
      }
      
    }
    
    void third_order_reverse_adj() {
      
      if (AutodiffOrder >= 3) {
        const double g3 = third_grad();
        third_grad(input()) += g3;
        third_grad(input(1)) -= g3;
        
        const double g4 = fourth_grad();
        fourth_grad(input()) += g4;
        fourth_grad(input(1)) -= g4;
      }
      
    }
    
  };
  
}

#endif
