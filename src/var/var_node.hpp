#ifndef nomad__src__var__var_node_hpp
#define nomad__src__var__var_node_hpp

#include <iostream>
#include <string>

#include <src/autodiff/autodiff_stack.hpp>

namespace nomad {
  
  class var_node_base {
  protected:
    
    nomad_idx_t dual_numbers_idx_;
    nomad_idx_t partials_idx_;
    nomad_idx_t inputs_idx_;
    
    nomad_idx_t n_inputs_;
    
  public:
    
    var_node_base(): n_inputs_(0) {

      dual_numbers_idx_ = next_dual_number_idx_;
      partials_idx_ = next_partials_idx_;
      inputs_idx_ = next_inputs_idx_;
      
      next_node_idx_++;
    
    }

    var_node_base(nomad_idx_t n_inputs):
    n_inputs_(n_inputs) {
      
      dual_numbers_idx_ = next_dual_number_idx_;
      partials_idx_ = next_partials_idx_;
      inputs_idx_ = next_inputs_idx_;
      
      next_node_idx_++;
    }
    
    virtual ~var_node_base();
    
    var_node_base& operator=(const var_node_base& rhs) {
      dual_numbers_idx_ = rhs.dual_numbers();
      partials_idx_ = rhs.partials();
      inputs_idx_ = rhs.inputs();
      n_inputs_ = n_inputs();
      return *this;
    }

    inline nomad_idx_t dual_numbers() const { return dual_numbers_idx_; }
    inline nomad_idx_t partials() const { return partials_idx_; }
    inline nomad_idx_t inputs() const { return inputs_idx_; }
    inline nomad_idx_t n_inputs() const { return n_inputs_; }
    
    nomad_idx_t input() { return inputs_[inputs_idx_]; }
    nomad_idx_t input(unsigned int k) { return inputs_[inputs_idx_ + k]; }
    
    constexpr static bool dynamic_inputs() { return true; }
    
    // Are these used?
    //nomad_idx_t begin() { return inputs_[inputs_idx_]; }
    //nomad_idx_t end() { return inputs_[inputs_idx_ + n_inputs_]; }
    
    inline double& first_val()   { return dual_numbers_[dual_numbers_idx_];     }
    inline double& first_grad()  { return dual_numbers_[dual_numbers_idx_ + 1]; }
    inline double& second_val()  { return dual_numbers_[dual_numbers_idx_ + 2]; }
    inline double& second_grad() { return dual_numbers_[dual_numbers_idx_ + 3]; }
    inline double& third_val()   { return dual_numbers_[dual_numbers_idx_ + 4]; }
    inline double& third_grad()  { return dual_numbers_[dual_numbers_idx_ + 5]; }
    inline double& fourth_val()  { return dual_numbers_[dual_numbers_idx_ + 6]; }
    inline double& fourth_grad() { return dual_numbers_[dual_numbers_idx_ + 7]; }
    
    inline static double& first_val(nomad_idx_t idx)   { return dual_numbers_[idx];     }
    inline static double& first_grad(nomad_idx_t idx)  { return dual_numbers_[idx + 1]; }
    inline static double& second_val(nomad_idx_t idx)  { return dual_numbers_[idx + 2]; }
    inline static double& second_grad(nomad_idx_t idx) { return dual_numbers_[idx + 3]; }
    inline static double& third_val(nomad_idx_t idx)   { return dual_numbers_[idx + 4]; }
    inline static double& third_grad(nomad_idx_t idx)  { return dual_numbers_[idx + 5]; }
    inline static double& fourth_val(nomad_idx_t idx)  { return dual_numbers_[idx + 6]; }
    inline static double& fourth_grad(nomad_idx_t idx) { return dual_numbers_[idx + 7]; }

    inline double* first_partials()  { return partials_ + partials_idx_; }
    inline double* second_partials() { return partials_ + partials_idx_ + n_first_partials(); }
    inline double* third_partials()  { return partials_ + partials_idx_ + n_first_partials() + n_second_partials(); }

    inline double first_partials(nomad_idx_t idx)  {
      return partials_[partials_idx_ + idx];
    }
    inline double second_partials(nomad_idx_t idx) {
      return partials_[partials_idx_ + n_first_partials() + idx];
    }
    inline double third_partials(nomad_idx_t idx)  {
      return partials_[partials_idx_ + n_first_partials() + n_second_partials() + idx];
    }
    
    inline virtual nomad_idx_t n_first_partials() { return 0; }
    inline virtual nomad_idx_t n_second_partials() { return 0; }
    inline virtual nomad_idx_t n_third_partials() { return 0; }
    
    inline static nomad_idx_t n_partials(unsigned int n_inputs) { (void)n_inputs; return 0; }

    template<short AutodiffOrder>
    void print(std::ostream* output, std::string prefix = "") {
      if(!output) return;
      
      *output << prefix << "Dual numbers stored at index " << dual_numbers_idx_ << std::endl;
      
      if (AutodiffOrder >= 1) {
        *output << prefix << "  first val = " << first_val()
                << ", first grad = " << first_grad() << std::endl;
      }
      if (AutodiffOrder >= 2) {
        *output << prefix << "  second val = " << second_val()
                << ", second grad = " << second_grad() << std::endl;
      }
      if (AutodiffOrder >= 2) {
        *output << prefix << "  third val = " << third_val()
                << ", third grad = " << third_grad() << std::endl;
        *output << prefix << "  fourth val = " << fourth_val()
                << ", fourth grad = " << fourth_grad() << std::endl;
      }
      *output << std::endl;
      
      if (n_inputs_) {
        *output << prefix << "Inputs stored at index " << inputs_idx_ << std::endl;
        *output << prefix << "Dual number index of inputs: " << std::endl;
        for (nomad_idx_t i = 0; i < n_inputs_; ++i)
          *output << prefix << "  (" << i << ") " << input(i) << std::endl;
        *output << std::endl;
      }
      
      if (n_partials(n_inputs_)) {
        *output << prefix << "Partials stored at index = " << partials_idx_ << std::endl;
        
        if (n_first_partials()) *output << prefix << "First partials: " << std::endl;
        for (nomad_idx_t i = 0; i < n_first_partials(); ++i)
          *output << prefix << "  (" << i << ") " << first_partials()[i] << std::endl;
        
        if (n_second_partials()) *output << prefix << "Second partials: " << std::endl;
        for (nomad_idx_t i = 0; i < n_second_partials(); ++i)
          *output << prefix << "  (" << i << ") " << second_partials()[i] << std::endl;
        
        if (n_third_partials()) *output << prefix << "Third partials: " << std::endl;
        for (nomad_idx_t i = 0; i < n_third_partials(); ++i)
          *output << prefix << "  (" << i << ") " << third_partials()[i] << std::endl;
        
        *output << std::endl;
      }
      
    }
    
    virtual void first_order_forward_adj()  {}
    virtual void first_order_reverse_adj()  {}
    virtual void second_order_forward_val() {}
    virtual void second_order_reverse_adj() {}
    virtual void third_order_forward_val()  {}
    virtual void third_order_reverse_adj()  {}
    
  };
  
  // Out-of-line virtual defintion to prevent weak vtable
  var_node_base::~var_node_base() {}
  
  template<short AutodiffOrder>
  void expand_var_nodes() {
    
    if (!max_node_idx) {
      max_node_idx = base_node_size_;
      var_nodes_ = new var_node_base[max_node_idx];
      next_node_idx_ -= max_node_idx;
    } else {
      max_node_idx *= 2;
      
      var_node_base* new_stack = new var_node_base[max_node_idx];
      for (nomad_idx_t i = 0; i < next_node_idx_; ++i)
        new_stack[i] = var_nodes_[i];
      delete[] var_nodes_;
      
      var_nodes_ = new_stack;
    }
    
    expand_dual_numbers<AutodiffOrder>();
    
  }
  
  template<short AutodiffOrder, short PartialsOrder>
  class var_node: public var_node_base {
  public:
    
    static inline void* operator new(size_t /* ignore */) {
      if (unlikely(next_node_idx_ + 1 > max_node_idx)) expand_var_nodes<AutodiffOrder>();
      if (unlikely(next_partials_idx_ + next_partials_delta > max_partials_idx)) expand_partials();
      if (unlikely(next_inputs_idx_ + next_inputs_delta > max_inputs_idx)) expand_inputs();
      return var_nodes_ + next_node_idx_;
    }
    
    static inline void operator delete(void* /* ignore */) {}
    
    var_node(nomad_idx_t n_inputs): var_node_base(n_inputs) {}

    inline nomad_idx_t n_first_partials() {
      return AutodiffOrder >= 1 && PartialsOrder >= 1 ?
             n_inputs_ : 0;
    }
    
    inline nomad_idx_t n_second_partials() {
      return AutodiffOrder >= 2 && PartialsOrder >= 2 ?
             n_inputs_ * (n_inputs_ + 1) / 2 : 0;
    }
    
    inline nomad_idx_t n_third_partials() {
      return AutodiffOrder >= 3 && PartialsOrder >= 3 ?
             n_inputs_ * (n_inputs_ + 1) * (n_inputs_ + 2) / 6 : 0;
    }
    
    inline static nomad_idx_t n_partials(unsigned int n_inputs) {
      if (AutodiffOrder >= 1 && PartialsOrder >= 1)
        return n_inputs;
      
      // n + n * (n + 1) / 2
      if (AutodiffOrder >= 2 && PartialsOrder >= 2)
        return n_inputs * (n_inputs + 3) / 2;
      
      // n + n * (n + 1) / 2 + n * (n + 1) * (n + 2) / 6
      if (AutodiffOrder >= 3 && PartialsOrder >= 3)
        return n_inputs * (11 + 6 * n_inputs + n_inputs * n_inputs) / 6;
      
      return 0;
    }
    
    inline void first_order_forward_adj() {
      
      if (n_inputs_) first_grad() = 0;
      
      if (PartialsOrder >= 1) {
        
        double* first_partial = first_partials();
        
        double g = 0;
        
        for (nomad_idx_t i = 0; i < n_inputs_; ++i, ++first_partial)
          g += first_grad(input(i)) * *first_partial;
        first_grad() += g;
        
      }
      
    }
    
    inline void first_order_reverse_adj() {
      if (PartialsOrder >= 1) {
        double* first_partial = first_partials();
        const double g = first_grad();
        for (nomad_idx_t i = 0; i < n_inputs_; ++i, ++first_partial)
          first_grad(input(i)) += g * *first_partial;
      }
      
    }
    
    void second_order_forward_val() {
      
      if (AutodiffOrder >= 2) {
        
        if (n_inputs_) second_val() = 0;
        second_grad() = 0;
        
        if (PartialsOrder >= 1) {
          
          double* first_partial = first_partials();
          double v2 = 0;
          
          for (nomad_idx_t i = 0; i < n_inputs_; ++i, ++first_partial)
            v2 += second_val(input(i)) * *first_partial;
          second_val() += v2;
          
        }
        
      }
      
    }
    
    void second_order_reverse_adj() {
      
      if (AutodiffOrder >= 2) {
        
        if (PartialsOrder >= 1) {
          
          double* first_partial = first_partials();
          const double g = second_grad();
          
          for (nomad_idx_t i = 0; i < n_inputs_; ++i, ++first_partial)
            second_grad(input(i)) += g * *first_partial;
          
        }
        
        if (PartialsOrder >= 2) {
          
          const double g1 = first_grad();
          
          for (nomad_idx_t i = 0; i < n_inputs_; ++i) {
            
            double g2 = 0;
            
            double* second_partial = second_partials() + i * (i + 1) / 2;
            for (nomad_idx_t j = 0; j < n_inputs_; ++j) {
              
              g2 += g1 * second_val(input(j)) * *second_partial;
              
              if (j < i) ++second_partial;
              else       second_partial += j + 1;
              
            }
            
            second_grad(input(i)) += g2;
            
          }
          
        }
        
      }
      
    }
    
    void third_order_forward_val() {
      
      if (AutodiffOrder >= 3) {
        
        if (n_inputs_) {
          third_val() = 0;
          fourth_val() = 0;
        }
        
        third_grad() = 0;
        fourth_grad() = 0;
        
        if (PartialsOrder >= 1) {
          
          double* first_partial = first_partials();
          double v3 = 0;
          double v4 = 0;
          
          for (nomad_idx_t i = 0; i < n_inputs_; ++i, ++first_partial) {
            v3 += third_val(input(i)) * *first_partial;
            v4 += fourth_val(input(i)) * *first_partial;
          }
          third_val() += v3;
          fourth_val() += v4;
          
        }
        
        if (PartialsOrder >= 2) {
          
          double v4 = 0;
          
          for (nomad_idx_t i = 0; i < n_inputs_; ++i) {
            
            const double s2 = second_val(input(i));
            
            // Benchmark in future
            //if(s2 == 0) continue;
            
            double* second_partial = second_partials() + i * (i + 1) / 2;
            for (nomad_idx_t j = 0; j < n_inputs_; ++j) {
              
              v4 += s2 * third_val(input(j)) * *second_partial;
              
              if (j < i) ++second_partial;
              else       second_partial += j + 1;
              
            }
          }

          fourth_val() += v4;
          
        }
        
      }
      
    } // third_order_forward_val
    
    void third_order_reverse_adj() {
      
      if (AutodiffOrder >= 3) {

        const double g1 = first_grad();
        const double g2 = second_grad();
        const double g3 = third_grad();
        const double g4 = fourth_grad();
        
        if (PartialsOrder >= 1) {
          
          double* first_partial = first_partials();
          
          for (nomad_idx_t i = 0; i < n_inputs_; ++i, ++first_partial) {
            third_grad(input(i))  += g3 * *first_partial;
            fourth_grad(input(i)) += g4 * *first_partial;
          }
        }
        
        if (PartialsOrder >= 2) {
          
          for (nomad_idx_t i = 0; i < n_inputs_; ++i) {
            
            double in_g3 = 0;
            double in_g4 = 0;
            
            double* second_partial = second_partials() + i * (i + 1) / 2;
            for (nomad_idx_t j = 0; j < n_inputs_; ++j) {
              
              /*
              // Benchmark in future
              if(*second_partial == 0) {
                if (j < i) ++second_partial;
                else       second_partial += j + 1;
                continue;
              }
              */
              
              in_g3 += g1 * third_val(input(j)) * *second_partial;
              
              double alpha =   g1 * fourth_val(input(j))
                             + g2 * third_val(input(j))
                             + g3 * second_val(input(j));
              
              in_g4 += alpha * *second_partial;
              
              if (j < i) ++second_partial;
              else       second_partial += j + 1;

            }

            third_grad(input(i)) += in_g3;
            fourth_grad(input(i)) += in_g4;
            
          }
          
        }
        
        if (PartialsOrder >= 3) {

          for (nomad_idx_t i = 0; i < n_inputs_; ++i) {
            
            double in_g4 = 0;
            
            for (nomad_idx_t j = 0; j < n_inputs_; ++j) {
              
              const double in_v2 = second_val(input(j));
              
              double* third_partial = third_partials();
              
              if (j < i)
                third_partial += + i * (i + 1) * (i + 2) / 6 + j * (j + 1) / 2; // Dense storage
              else
                third_partial += + i * (i + 1) / 2 + j * (j + 1) * (j + 2) / 6; // Sparse storage
              
              for (nomad_idx_t k = 0; k < n_inputs_; ++k) {

                in_g4 +=   g1
                         * in_v2
                         * third_val(input(k))
                         * *third_partial;
 
                if (k < i) {
                  if (k < j) ++third_partial;                        // Dense-Dense storage
                  else       third_partial += k + 1;                 // Dense-Sparse storage
                }
                else {
                  if (k < j) third_partial += k + 1;                 // Dense-Sparse Storage
                  else       third_partial += (k + 1) * (k + 2) / 2; // Sparse-Sparse storage
                }
                
              }
              
            }
            
            fourth_grad(input(i)) += in_g4;
            
          }
          
        }
        
      }
      
    } // third_order_reverse_adj
    
  };
  
  template<short AutodiffOrder>
  class var_node<AutodiffOrder, 0>: public var_node_base {
    
  public:
    
    static inline void* operator new(size_t /* ignore */) {
      if (unlikely(next_node_idx_ + 1 > max_node_idx)) expand_var_nodes<AutodiffOrder>();
      // no partials
      // no inputs
      return var_nodes_ + next_node_idx_;
    }
    
    static inline void operator delete(void* /* ignore */) {}
    
    var_node(): var_node_base() {}
    
    constexpr static bool dynamic_inputs() { return false; }
    inline static nomad_idx_t n_partials() { return 0; }
    
    void second_order_forward_val() {
      second_grad() = 0;
    }
    
    void third_order_forward_val() {
      third_grad() = 0;
      fourth_grad() = 0;
    }
    
  };
  
}

#endif
