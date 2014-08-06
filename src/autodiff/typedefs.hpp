#ifndef nomad__src__autodiff__typedefs_hpp
#define nomad__src__autodiff__typedefs_hpp

#include <Eigen/Core>

namespace nomad {
  
  typedef unsigned int nomad_idx_t;
  typedef Eigen::internal::traits<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>::Index
    eigen_idx_t;
}

#endif
