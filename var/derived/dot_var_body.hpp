#ifndef nomad__var__derived__dot_var_body_hpp
#define nomad__var__derived__dot_var_body_hpp

#include <var/var_body.hpp>

namespace nomad {

  template<short autodiff_order>
  class dot_var_body: public var_base {
  public:
    
    static inline void* operator new(size_t /* ignore */) {
      if (unlikely(next_body_idx_ + 1 > max_body_idx)) expand_var_bodies<autodiff_order>();
      // no partials
      if (unlikely(next_inputs_idx_ + next_inputs_delta > max_inputs_idx)) expand_inputs();
      return var_bodies_ + next_body_idx_;
    }
    
    static inline void operator delete(void* /* ignore */) {};
    
    dot_var_body(int n_inputs): var_base(n_inputs) {};
 
    inline unsigned int n_first_partials() { return 0; }
    inline unsigned int n_second_partials() { return 0; }
    inline unsigned int n_third_partials() { return 0; }
    inline static unsigned int n_partials(unsigned int n_inputs) { return 0; }
    
    inline void first_order_forward_adj() {
      
      if (autodiff_order >= 1) {
        
        if (n_inputs_) first_grad() = 0;
        
        double g1 = 0;
        
        unsigned int end = n_inputs_ / 2;
        
        for (unsigned int i = 0; i < end; ++i)
          g1 += first_grad(input(i)) * first_val(input(i + end));
        
        for (unsigned int i = 0; i < end; ++i)
          g1 += first_grad(input(i + end)) * first_val(input(i));
        
        first_grad() += g1;
        
      }
      
    }
    
    inline void first_order_reverse_adj() {
      
      if (autodiff_order >= 1) {
        
        unsigned int end = n_inputs_ / 2;
        
        const double g1 = first_grad();
        
        for (unsigned int i = 0; i < end; ++i) {
          first_grad(input(i)) += g1 * first_val(input(i + end));
          first_grad(input(i + end)) += g1 * first_val(input(i));
        }
      }
    }
    
    void second_order_forward_val() {
      
      if (autodiff_order >= 2) {
        
        if (n_inputs_) second_val() = 0;
        second_grad() = 0;
        
        unsigned int end = n_inputs_ / 2;
        
        double v2 = 0;
        
        for (unsigned int i = 0; i < end; ++i)
          v2 += second_val(input(i)) * first_val(input(i + end));
        
        for (unsigned int i = 0; i < end; ++i)
          v2 += second_val(input(i + end)) * first_val(input(i));
        
        second_val() += v2;
        
      }
    }
    
    void second_order_reverse_adj() {
      
      if (autodiff_order >= 2) {
        
        unsigned int end = n_inputs_ / 2;
        
        const double g1 = first_grad();
        const double g2 = second_grad();
        
        for (unsigned int i = 0; i < end; ++i)
          second_grad(input(i)) +=  g2 * first_val(input(i + end))
                                  + g1 * second_val(input(i + end));
        
        for (unsigned int i = 0; i < end; ++i)
          second_grad(input(i + end)) +=  g2 * first_val(input(i))
                                        + g1 * second_val(input(i));
        
      }
      
    }
    
    void third_order_forward_val() {
      
      if (autodiff_order >= 3) {
        
        if (n_inputs_) {
          third_val() = 0;
          fourth_val() = 0;
        }
        
        double v3 = 0;
        double v4 = 0;
        
        unsigned int end = n_inputs_ / 2;
        
        for (unsigned int i = 0; i < end; ++i) {
          v3 += third_val(input(i)) * first_val(input(i + end));
          v4 +=  fourth_val(input(i)) * first_val(input(i + end))
               + second_val(input(i)) * third_val(input(i + end));
        }
        
        for (unsigned int i = 0; i < end; ++i) {
          v3 += third_val(input(i + end)) * first_val(input(i));
          v4 +=  fourth_val(input(i + end)) * first_val(input(i))
               + second_val(input(i + end)) * third_val(input(i));
        }
        
        third_val() += v3;
        fourth_val() += v4;
        
      }
      
    } // third_order_forward_val
    
    void third_order_reverse_adj() {
      
      if (autodiff_order >= 3) {
        
        unsigned int end = n_inputs_ / 2;
        
        const double g1 = first_grad();
        const double g2 = second_grad();
        const double g3 = third_grad();
        const double g4 = fourth_grad();
        
        for (unsigned int i = 0; i < end; ++i) {
          third_grad(input(i))  += g3 * first_val(input(i + end)) + g1 * third_val(input(i + end));
          fourth_grad(input(i)) +=   g4 * first_val(input(i + end))
                                   + g3 * second_val(input(i + end))
                                   + g2 * third_val(input(i + end))
                                   + g1 * fourth_val(input(i + end));
        }
        
        for (unsigned int i = 0; i < end; ++i) {
          third_grad(input(i + end))  += g3 * first_val(input(i)) + g1 * third_val(input(i));
          fourth_grad(input(i + end)) +=   g4 * first_val(input(i))
                                         + g3 * second_val(input(i))
                                         + g2 * third_val(input(i))
                                         + g1 * fourth_val(input(i));
        }
        
      }
      
    } // third_order_reverse_adj
    
  };
  
}

#endif
