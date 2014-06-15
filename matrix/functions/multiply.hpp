#ifndef nomad__matrix__functions__multiply_hpp
#define nomad__matrix__functions__multiply_hpp

#include <Eigen/Core>

#include <matrix/functions/dot.hpp>
#include <var/var.hpp>

namespace nomad {
  
  template<typename DerivedA, typename DerivedB>
  inline typename std::enable_if<
    is_var<typename Eigen::MatrixBase<DerivedA>::Scalar>::value &&
    is_var<typename Eigen::MatrixBase<DerivedB>::Scalar>::value &&
    std::is_same<typename Eigen::MatrixBase<DerivedA>::Scalar,
                 typename Eigen::MatrixBase<DerivedB>::Scalar>::value,
    typename Eigen::Matrix<var<Eigen::MatrixBase<DerivedA>::Scalar::order()>,
             Eigen::Dynamic, Eigen::Dynamic>
    >::type
  multiply(const Eigen::MatrixBase<DerivedA>& M1,
           const Eigen::MatrixBase<DerivedB>& M2) {
    
    const int N = M1.cols();
    Eigen::Matrix<var<Eigen::MatrixBase<DerivedA>::Scalar::order()>, Eigen::Dynamic, Eigen::Dynamic> M(N, N);
    
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        M(i, j) = dot(M1.row(i), M2.col(j));
      }
    }
    
    return M;
    
  }

}

#endif
