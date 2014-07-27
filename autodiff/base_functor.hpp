#ifndef nomad__autodiff__base_functor_hpp
#define nomad__autodiff__base_functor_hpp

#include <Eigen/Core>

namespace nomad {
  
  template<class T>
  class base_functor {
  public:
    virtual T operator()(const Eigen::VectorXd& x) const;
    typedef T var_type;
  };
  
}

#endif
