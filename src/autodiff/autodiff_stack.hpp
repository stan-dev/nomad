#ifndef nomad__src__autodiff__autodiff_stack_hpp
#define nomad__src__autodiff__autodiff_stack_hpp

#ifdef __GNUC__
#define likely(x) (__builtin_expect((x),1))
#define unlikely(x) (__builtin_expect((x),0))
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif

#include <vector>
#include <type_traits>

#include <src/autodiff/typedefs.hpp>
#include <src/autodiff/exceptions.hpp>

namespace nomad {
  
  void reset();
  void expand_var_nodes();
  template<short AutodiffOrder> void expand_dual_numbers();
  void expand_partials();
  void expand_inputs();

  class var_node_base;
  
  nomad_idx_t base_node_size_ = 100000;
  
  var_node_base* var_nodes_;
  nomad_idx_t next_node_idx_ = 1;
  nomad_idx_t max_node_idx = 0;
  
  double* dual_numbers_;
  nomad_idx_t next_dual_number_idx_ = 1;
  nomad_idx_t max_dual_number_idx = 0;
  
  nomad_idx_t base_partials_size_ = 100000;
  
  double* partials_;
  nomad_idx_t next_partials_idx_ = 1;
  nomad_idx_t next_partials_delta = 0;
  nomad_idx_t max_partials_idx = 0;
  
  nomad_idx_t base_inputs_size_ = 100000;
  
  nomad_idx_t* inputs_;
  nomad_idx_t next_inputs_idx_ = 1;
  nomad_idx_t next_inputs_delta = 0;
  nomad_idx_t max_inputs_idx = 0;
  
  void reset() {
    next_node_idx_ = 1;
    next_dual_number_idx_ = 1;
    next_partials_idx_ = 1;
    next_inputs_idx_ = 1;
  }
  
  template <class Node>
  inline typename std::enable_if<Node::dynamic_inputs(), void>::type create_node(unsigned int n_inputs) {
    next_inputs_delta = n_inputs;
    next_partials_delta = Node::n_partials(n_inputs);
    new Node(n_inputs);
  }
  
  template <typename Node>
  inline typename std::enable_if<!Node::dynamic_inputs(), void>::type create_node(unsigned int n_inputs) {
    next_inputs_delta = n_inputs;
    next_partials_delta = Node::n_partials();
    new Node();
  }
  
  template<short AutodiffOrder, bool ValidateIO>
  inline void push_dual_numbers(double val) {
    
    if (ValidateIO) {
      if (unlikely(std::isnan(val))) throw nomad_error();
    }

    dual_numbers_[next_dual_number_idx_++] = val;
    dual_numbers_[next_dual_number_idx_++] = 0;
    
    if (AutodiffOrder >= 2) {
      dual_numbers_[next_dual_number_idx_++] = 0;
      dual_numbers_[next_dual_number_idx_++] = 0;
    }
    
    if (AutodiffOrder >= 3) {
      dual_numbers_[next_dual_number_idx_++] = 0;
      dual_numbers_[next_dual_number_idx_++] = 0;
      dual_numbers_[next_dual_number_idx_++] = 0;
      dual_numbers_[next_dual_number_idx_++] = 0;
    }
    
  }
  
  template<bool ValidateIO>
  inline void push_partials(double partial) {
    if (ValidateIO) {
      if (unlikely(std::isnan(partial))) throw nomad_error();
    }
    partials_[next_partials_idx_++] = partial;
  }
  
  inline void push_inputs(nomad_idx_t input) {
    inputs_[next_inputs_idx_++] = input;
  }
  
  void expand_var_nodes();
  
  template<short AutodiffOrder>
  inline void expand_dual_numbers() {
    
    if (!max_dual_number_idx) {
      max_dual_number_idx = (1 << AutodiffOrder) * base_node_size_;
      dual_numbers_ = new double[max_dual_number_idx];
    } else {
      max_dual_number_idx *= 2;
      
      double* new_stack = new double[max_dual_number_idx];
      for (nomad_idx_t i = 0; i < next_dual_number_idx_; ++i)
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
      for (nomad_idx_t i = 0; i < next_partials_idx_; ++i)
        new_stack[i] = partials_[i];
      delete[] partials_;
      
      partials_ = new_stack;
    }
    
  }
  
  void expand_inputs() {
    if (!max_inputs_idx) {
      max_inputs_idx = base_inputs_size_;
      inputs_ = new nomad_idx_t[max_inputs_idx];
    } else {
      max_inputs_idx *= 2;
      
      nomad_idx_t* new_stack = new nomad_idx_t[max_inputs_idx];
      for (nomad_idx_t i = 0; i < next_inputs_idx_; ++i)
        new_stack[i] = inputs_[i];
      delete[] inputs_;
      
      inputs_ = new_stack;
    }
    
  }
  
}

#endif
