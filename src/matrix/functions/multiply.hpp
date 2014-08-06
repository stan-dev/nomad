#ifndef nomad__src__matrix__functions__multiply_hpp
#define nomad__src__matrix__functions__multiply_hpp

#include <Eigen/Core>

#include <src/matrix/functions/dot.hpp>
#include <src/var/var.hpp>

namespace nomad {
  
  template<typename DerivedA, typename DerivedB>
  inline typename std::enable_if<
    is_var<typename Eigen::MatrixBase<DerivedA>::Scalar>::value &&
    is_var<typename Eigen::MatrixBase<DerivedB>::Scalar>::value &&
    std::is_same<typename Eigen::MatrixBase<DerivedA>::Scalar,
                 typename Eigen::MatrixBase<DerivedB>::Scalar>::value,
    typename Eigen::Matrix<typename Eigen::MatrixBase<DerivedA>::Scalar,
             Eigen::Dynamic, Eigen::Dynamic>
    >::type
  multiply(const Eigen::MatrixBase<DerivedA>& M1,
           const Eigen::MatrixBase<DerivedB>& M2) {
    
    const eigen_idx_t N = M1.cols();
    Eigen::Matrix<typename Eigen::MatrixBase<DerivedA>::Scalar, Eigen::Dynamic, Eigen::Dynamic> M(N, N);
    
    for (eigen_idx_t i = 0; i < N; ++i) {
      for (eigen_idx_t j = 0; j < N; ++j) {
        M(i, j) = dot(M1.row(i), M2.col(j));
      }
    }
    
    return M;
    
  }

}

#endif
