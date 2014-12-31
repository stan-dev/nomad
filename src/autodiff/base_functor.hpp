#ifndef nomad__src__autodiff__base_functor_hpp
#define nomad__src__autodiff__base_functor_hpp

#include <Eigen/Core>

namespace nomad {
  
  template<class T>
  class base_functor {
  public:
    virtual ~base_functor() {}
    virtual T operator()(const Eigen::VectorXd& x) const;
    typedef T var_type;
  };
  
}

#endif
