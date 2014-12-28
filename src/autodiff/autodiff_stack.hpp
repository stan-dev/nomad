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
  
  // Wrap global variables in a templated struct to ensure
  // only a single declaration in each translation unit
  
  template <typename NodeBase>
  struct nomad_storage {
    static nomad_idx_t base_node_size;
    
    static NodeBase* var_nodes;
    static nomad_idx_t next_node_idx;
    static nomad_idx_t max_node_idx;
    
    static double* dual_numbers;
    static nomad_idx_t next_dual_number_idx;
    static nomad_idx_t max_dual_number_idx;
    
    static nomad_idx_t base_partials_size;
    
    static double* partials;
    static nomad_idx_t next_partials_idx;
    static nomad_idx_t next_partials_delta;
    static nomad_idx_t max_partials_idx;
    
    static nomad_idx_t base_inputs_size;
    
    static nomad_idx_t* inputs;
    static nomad_idx_t next_inputs_idx;
    static nomad_idx_t next_inputs_delta;
    static nomad_idx_t max_inputs_idx;
  };
  
  template <typename NodeBase> nomad_idx_t nomad_storage<NodeBase>::base_node_size = 100000;
  
  template <typename NodeBase> NodeBase* nomad_storage<NodeBase>::var_nodes = nullptr;
  template <typename NodeBase> nomad_idx_t nomad_storage<NodeBase>::next_node_idx = 1;
  template <typename NodeBase> nomad_idx_t nomad_storage<NodeBase>::max_node_idx = 0;
  
  template <typename NodeBase> double* nomad_storage<NodeBase>::dual_numbers = nullptr;
  template <typename NodeBase> nomad_idx_t nomad_storage<NodeBase>::next_dual_number_idx = 1;
  template <typename NodeBase> nomad_idx_t nomad_storage<NodeBase>::max_dual_number_idx = 0;

  template <typename NodeBase> nomad_idx_t nomad_storage<NodeBase>::base_partials_size = 100000;
  
  template <typename NodeBase> double* nomad_storage<NodeBase>::partials = nullptr;
  template <typename NodeBase> nomad_idx_t nomad_storage<NodeBase>::next_partials_idx = 1;
  template <typename NodeBase> nomad_idx_t nomad_storage<NodeBase>::next_partials_delta = 0;
  template <typename NodeBase> nomad_idx_t nomad_storage<NodeBase>::max_partials_idx = 0;
  
  template <typename NodeBase> nomad_idx_t nomad_storage<NodeBase>::base_inputs_size = 100000;
  
  template <typename NodeBase> nomad_idx_t* nomad_storage<NodeBase>::inputs = nullptr;
  template <typename NodeBase> nomad_idx_t nomad_storage<NodeBase>::next_inputs_idx = 1;
  template <typename NodeBase> nomad_idx_t nomad_storage<NodeBase>::next_inputs_delta = 0;
  template <typename NodeBase> nomad_idx_t nomad_storage<NodeBase>::max_inputs_idx = 0;
  
  class var_node_base;
  typedef nomad_storage<var_node_base> nmd_stk;
  
  // Stack manipulation functions
  static inline void reset() {
    nmd_stk::next_node_idx = 1;
    nmd_stk::next_dual_number_idx = 1;
    nmd_stk::next_partials_idx = 1;
    nmd_stk::next_inputs_idx = 1;
  }
  
  template <class Node>
  static inline typename std::enable_if<Node::dynamic_inputs(), void>::type
  create_node(unsigned int n_inputs) {
    nmd_stk::next_inputs_delta = n_inputs;
    nmd_stk::next_partials_delta = Node::n_partials(n_inputs);
    new Node(n_inputs);
  }
  
  template <typename Node>
  static inline typename std::enable_if<!Node::dynamic_inputs(), void>::type
  create_node(unsigned int n_inputs) {
    nmd_stk::next_inputs_delta = n_inputs;
    nmd_stk::next_partials_delta = Node::n_partials();
    new Node();
  }
  
  template<short AutodiffOrder, bool ValidateIO>
  static inline void push_dual_numbers(double val) {
    
    if (ValidateIO) {
      if (unlikely(std::isnan(val))) throw nomad_error();
    }

    nmd_stk::dual_numbers[nmd_stk::next_dual_number_idx++] = val;
    nmd_stk::dual_numbers[nmd_stk::next_dual_number_idx++] = 0;
    
    if (AutodiffOrder >= 2) {
      nmd_stk::dual_numbers[nmd_stk::next_dual_number_idx++] = 0;
      nmd_stk::dual_numbers[nmd_stk::next_dual_number_idx++] = 0;
    }
    
    if (AutodiffOrder >= 3) {
      nmd_stk::dual_numbers[nmd_stk::next_dual_number_idx++] = 0;
      nmd_stk::dual_numbers[nmd_stk::next_dual_number_idx++] = 0;
      nmd_stk::dual_numbers[nmd_stk::next_dual_number_idx++] = 0;
      nmd_stk::dual_numbers[nmd_stk::next_dual_number_idx++] = 0;
    }
    
  }
  
  template<bool ValidateIO>
  static inline void push_partials(double partial) {
    if (ValidateIO) {
      if (unlikely(std::isnan(partial))) throw nomad_error();
    }
    nmd_stk::partials[nmd_stk::next_partials_idx++] = partial;
  }
  
  static inline void push_inputs(nomad_idx_t input) {
    nmd_stk::inputs[nmd_stk::next_inputs_idx++] = input;
  }
  
  template<short AutodiffOrder>
  static inline void expand_dual_numbers() {
    
    if (!nmd_stk::max_dual_number_idx) {
      nmd_stk::max_dual_number_idx = (1 << AutodiffOrder) * nmd_stk::base_node_size;
      nmd_stk::dual_numbers = new double[nmd_stk::max_dual_number_idx];
    } else {
      nmd_stk::max_dual_number_idx *= 2;
      
      double* new_stack = new double[nmd_stk::max_dual_number_idx];
      for (nomad_idx_t i = 0; i < nmd_stk::next_dual_number_idx; ++i)
        new_stack[i] = nmd_stk::dual_numbers[i];
      delete[] nmd_stk::dual_numbers;
      
      nmd_stk::dual_numbers = new_stack;
    }

  }
  
  static inline void expand_partials() {
    
    if (!nmd_stk::max_partials_idx) {
      nmd_stk::max_partials_idx = nmd_stk::base_partials_size;
      nmd_stk::partials = new double[nmd_stk::max_partials_idx];
    } else {
      nmd_stk::max_partials_idx *= 2;
      
      double* new_stack = new double[nmd_stk::max_partials_idx];
      for (nomad_idx_t i = 0; i < nmd_stk::next_partials_idx; ++i)
        new_stack[i] = nmd_stk::partials[i];
      delete[] nmd_stk::partials;
      
      nmd_stk::partials = new_stack;
    }
    
  }
  
  static inline void expand_inputs() {
    if (!nmd_stk::max_inputs_idx) {
      nmd_stk::max_inputs_idx = nmd_stk::base_inputs_size;
      nmd_stk::inputs = new nomad_idx_t[nmd_stk::max_inputs_idx];
    } else {
      nmd_stk::max_inputs_idx *= 2;
      
      nomad_idx_t* new_stack = new nomad_idx_t[nmd_stk::max_inputs_idx];
      for (nomad_idx_t i = 0; i < nmd_stk::next_inputs_idx; ++i)
        new_stack[i] = nmd_stk::inputs[i];
      delete[] nmd_stk::inputs;
      
      nmd_stk::inputs = new_stack;
    }
    
  }
  
}

#endif
