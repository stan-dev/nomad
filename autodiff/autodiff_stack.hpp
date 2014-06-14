#ifndef nomad__autodiff__autodiff_stack_hpp
#define nomad__autodiff__autodiff_stack_hpp

#ifdef __GNUC__
#define likely(x) (__builtin_expect((x),1))
#define unlikely(x) (__builtin_expect((x),0))
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif

#include <vector>

namespace nomad {

  typedef int index_t;
  
  class var_base;
  
  index_t base_body_size_ = 100000;
  
  var_base* var_bodies_;
  index_t next_body_idx_ = 1;
  index_t max_body_idx = 0;
  
  double* dual_numbers_;
  index_t next_dual_number_idx_ = 1;
  index_t max_dual_number_idx = 0;
  
  index_t base_partials_size_ = 100000;
  
  double* partials_;
  index_t next_partials_idx_ = 1;
  index_t next_partials_delta = 0;
  index_t max_partials_idx = 0;
  
  index_t base_inputs_size_ = 100000;
  
  index_t* inputs_;
  index_t next_inputs_idx_ = 1;
  index_t next_inputs_delta = 0;
  index_t max_inputs_idx = 0;
  
  void reset() {
    next_body_idx_ = 1;
    next_dual_number_idx_ = 1;
    next_partials_idx_ = 1;
    next_inputs_idx_ = 1;
  }
  
  template<short autodiff_order>
  inline void push_dual_numbers(double val) {
    
    dual_numbers_[next_dual_number_idx_++] = val;
    dual_numbers_[next_dual_number_idx_++] = 0;
    
    if (autodiff_order >= 2) {
      dual_numbers_[next_dual_number_idx_++] = 0;
      dual_numbers_[next_dual_number_idx_++] = 0;
    }
    
    if (autodiff_order >= 3) {
      dual_numbers_[next_dual_number_idx_++] = 0;
      dual_numbers_[next_dual_number_idx_++] = 0;
      dual_numbers_[next_dual_number_idx_++] = 0;
      dual_numbers_[next_dual_number_idx_++] = 0;
    }
    
  }
  
  inline void push_partials(double partial) {
    partials_[next_partials_idx_++] = partial;
  }
  
  inline void push_inputs(index_t input) {
    inputs_[next_inputs_idx_++] = input;
  }
  
  void expand_var_bodies();
  
  template<short autodiff_order>
  inline void expand_dual_numbers() {
    
    if (!max_dual_number_idx) {
      max_dual_number_idx = (1 << autodiff_order) * base_body_size_;
      dual_numbers_ = new double[max_dual_number_idx];
    } else {
      max_dual_number_idx *= 2;
      
      double* new_stack = new double[max_dual_number_idx];
      for (int i = 0; i < next_dual_number_idx_; ++i)
        new_stack[i] = dual_numbers_[i];
      delete[] dual_numbers_;
      
      dual_numbers_ = new_stack;
    }

  }
  
  void expand_partials() {
    
    if (!max_partials_idx) {
      max_partials_idx = base_partials_size_;
      partials_ = new double[max_partials_idx];
    } else {
      max_partials_idx *= 2;
      
      double* new_stack = new double[max_partials_idx];
      for (int i = 0; i < next_partials_idx_; ++i)
        new_stack[i] = partials_[i];
      delete[] partials_;
      
      partials_ = new_stack;
    }
    
  }
  
  void expand_inputs() {
    if (!max_inputs_idx) {
      max_inputs_idx = base_inputs_size_;
      inputs_ = new index_t[max_inputs_idx];
    } else {
      max_inputs_idx *= 2;
      
      index_t* new_stack = new index_t[max_inputs_idx];
      for (int i = 0; i < next_inputs_idx_; ++i)
        new_stack[i] = inputs_[i];
      delete[] inputs_;
      
      inputs_ = new_stack;
    }
    
  }
  
}

#endif
