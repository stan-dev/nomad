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
  
  template <typename T_idx>
  struct base_node_size_struct {
    static T_idx value;
  };
  template <typename T_idx> T_idx base_node_size_struct<T_idx>::value = 100000;
  
  typedef base_node_size_struct<nomad_idx_t> base_node_size;
  
  template <typename NodeBase>
  struct var_nodes_struct {
    static NodeBase* address;
  };
  template <typename NodeBase> NodeBase* var_nodes_struct<NodeBase>::address = nullptr;
  
  class var_node_base;
  typedef var_nodes_struct<var_node_base> var_nodes;
  
  template <typename T_idx>
  struct next_node_idx_struct {
    static T_idx value;
  };
  template <typename T_idx> T_idx next_node_idx_struct<T_idx>::value = 1;
  
  typedef next_node_idx_struct<nomad_idx_t> next_node_idx;
  
  template <typename T_idx>
  struct max_node_idx_struct {
    static T_idx value;
  };
  template <typename T_idx> T_idx max_node_idx_struct<T_idx>::value = 0;
  
  typedef max_node_idx_struct<nomad_idx_t> max_node_idx;
  
  template <typename T>
  struct dual_numbers_struct {
    static T* address;
  };
  template <typename T> T* dual_numbers_struct<T>::address = nullptr;
  
  typedef dual_numbers_struct<double> dual_numbers;
  
  template <typename T_idx>
  struct next_dual_numbers_idx_struct {
    static T_idx value;
  };
  template <typename T_idx> T_idx next_dual_numbers_idx_struct<T_idx>::value = 1;
  
  typedef next_dual_numbers_idx_struct<nomad_idx_t> next_dual_numbers_idx;
  
  template <typename T_idx>
  struct max_dual_numbers_idx_struct {
    static T_idx value;
  };
  template <typename T_idx> T_idx max_dual_numbers_idx_struct<T_idx>::value = 0;
  
  typedef max_dual_numbers_idx_struct<nomad_idx_t> max_dual_numbers_idx;
  
  template <typename T_idx>
  struct base_partials_size_struct {
    static T_idx value;
  };
  template <typename T_idx> T_idx base_partials_size_struct<T_idx>::value = 100000;
  
  typedef base_partials_size_struct<nomad_idx_t> base_partials_size;
  
  template <typename T>
  struct partials_struct {
    static T* address;
  };
  template <typename T> T* partials_struct<T>::address = nullptr;
  
  typedef partials_struct<double> partials;
  
  template <typename T_idx>
  struct next_partials_idx_struct {
    static T_idx value;
  };
  template <typename T_idx> T_idx next_partials_idx_struct<T_idx>::value = 1;
  
  typedef next_partials_idx_struct<nomad_idx_t> next_partials_idx;

  template <typename T_idx>
  struct next_partials_delta_struct {
    static T_idx value;
  };
  template <typename T_idx> T_idx next_partials_delta_struct<T_idx>::value = 0;
  
  typedef next_partials_delta_struct<nomad_idx_t> next_partials_delta;
  
  template <typename T_idx>
  struct max_partials_idx_struct {
    static T_idx value;
  };
  template <typename T_idx> T_idx max_partials_idx_struct<T_idx>::value = 0;
  
  typedef max_partials_idx_struct<nomad_idx_t> max_partials_idx;
  
  template <typename T_idx>
  struct base_inputs_size_struct {
    static T_idx value;
  };
  template <typename T_idx> T_idx base_inputs_size_struct<T_idx>::value = 100000;
  
  typedef base_inputs_size_struct<nomad_idx_t> base_inputs_size;
  
  template <typename T>
  struct inputs_struct {
    static T* address;
  };
  template <typename T> T* inputs_struct<T>::address = nullptr;
  
  typedef inputs_struct<nomad_idx_t> inputs;
  
  template <typename T_idx>
  struct next_inputs_idx_struct {
    static T_idx value;
  };
  template <typename T_idx> T_idx next_inputs_idx_struct<T_idx>::value = 1;
  
  typedef next_inputs_idx_struct<nomad_idx_t> next_inputs_idx;
  
  template <typename T_idx>
  struct next_inputs_delta_struct {
    static T_idx value;
  };
  template <typename T_idx> T_idx next_inputs_delta_struct<T_idx>::value = 0;
  
  typedef next_inputs_delta_struct<nomad_idx_t> next_inputs_delta;
  
  template <typename T_idx>
  struct max_inputs_idx_struct {
    static T_idx value;
  };
  template <typename T_idx> T_idx max_inputs_idx_struct<T_idx>::value = 0;
  
  typedef max_inputs_idx_struct<nomad_idx_t> max_inputs_idx;
  
  // Stack manipulation functions
  static inline void reset() {
    next_node_idx::value = 1;
    next_dual_numbers_idx::value = 1;
    next_partials_idx::value = 1;
    next_inputs_idx::value = 1;
  }
  
  template <class Node>
  static inline typename std::enable_if<Node::dynamic_inputs(), void>::type
  create_node(unsigned int n_inputs) {
    next_inputs_delta::value = n_inputs;
    next_partials_delta::value = Node::n_partials(n_inputs);
    new Node(n_inputs);
  }
  
  template <typename Node>
  static inline typename std::enable_if<!Node::dynamic_inputs(), void>::type
  create_node(unsigned int n_inputs) {
    next_inputs_delta::value = n_inputs;
    next_partials_delta::value = Node::n_partials();
    new Node();
  }
  
  template<short AutodiffOrder, bool ValidateIO>
  static inline void push_dual_numbers(double val) {
    
    if (ValidateIO) {
      if (unlikely(std::isnan(val))) throw nomad_error();
    }

    dual_numbers::address[next_dual_numbers_idx::value++] = val;
    dual_numbers::address[next_dual_numbers_idx::value++] = 0;
    
    if (AutodiffOrder >= 2) {
      dual_numbers::address[next_dual_numbers_idx::value++] = 0;
      dual_numbers::address[next_dual_numbers_idx::value++] = 0;
    }
    
    if (AutodiffOrder >= 3) {
      dual_numbers::address[next_dual_numbers_idx::value++] = 0;
      dual_numbers::address[next_dual_numbers_idx::value++] = 0;
      dual_numbers::address[next_dual_numbers_idx::value++] = 0;
      dual_numbers::address[next_dual_numbers_idx::value++] = 0;
    }
    
  }
  
  template<bool ValidateIO>
  static inline void push_partials(double partial) {
    if (ValidateIO) {
      if (unlikely(std::isnan(partial))) throw nomad_error();
    }
    partials::address[next_partials_idx::value++] = partial;
  }
  
  static inline void push_inputs(nomad_idx_t input) {
    inputs::address[next_inputs_idx::value++] = input;
  }
  
  template<short AutodiffOrder>
  static inline void expand_dual_numbers() {
    
    if (!max_dual_numbers_idx::value) {
      max_dual_numbers_idx::value = (1 << AutodiffOrder) * base_node_size::value;
      dual_numbers::address = new double[max_dual_numbers_idx::value];
    } else {
      max_dual_numbers_idx::value *= 2;
      
      double* new_stack = new double[max_dual_numbers_idx::value];
      for (nomad_idx_t i = 0; i < next_dual_numbers_idx::value; ++i)
        new_stack[i] = dual_numbers::address[i];
      delete[] dual_numbers::address;
      
      dual_numbers::address = new_stack;
    }

  }
  
  static inline void expand_partials() {
    
    if (!max_partials_idx::value) {
      max_partials_idx::value = base_partials_size::value;
      partials::address = new double[max_partials_idx::value];
    } else {
      max_partials_idx::value *= 2;
      
      double* new_stack = new double[max_partials_idx::value];
      for (nomad_idx_t i = 0; i < next_partials_idx::value; ++i)
        new_stack[i] = partials::address[i];
      delete[] partials::address;
      
      partials::address = new_stack;
    }
    
  }
  
  static inline void expand_inputs() {
    if (!max_inputs_idx::value) {
      max_inputs_idx::value = base_inputs_size::value;
      inputs::address = new nomad_idx_t[max_inputs_idx::value];
    } else {
      max_inputs_idx::value *= 2;
      
      nomad_idx_t* new_stack = new nomad_idx_t[max_inputs_idx::value];
      for (nomad_idx_t i = 0; i < next_inputs_idx::value; ++i)
        new_stack[i] = inputs::address[i];
      delete[] inputs::address;
      
      inputs::address = new_stack;
    }
    
  }
  
}

#endif
