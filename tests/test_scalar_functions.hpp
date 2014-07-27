#ifndef nomad__tests__test_scalar_functions_hpp
#define nomad__tests__test_scalar_functions_hpp

#include <string>

#include <autodiff/base_functor.hpp>
#include <scalar/functions.hpp>
#include <tests/finite_difference.hpp>

namespace nomad {

  // Prototypes
  void test_acos();
  void test_acosh();
  void test_asin();
  void test_asinh();
  void test_atan();
  void test_atan2();
  void test_atanh();
  void test_binary_prod_cubes();
  void test_cbrt();
  void test_cos();
  void test_cosh();
  void test_erf();
  void test_erfc();
  void test_exp();
  void test_exp2();
  void test_expm1();
  void test_fma();
  void test_hypot();
  void test_inv_cloglog();
  void test_inv_logit();
  void test_inv_sqrt();
  void test_inv_square();
  void test_inv();
  void test_lgamma();
  void test_log_diff_exp();
  void test_log_sum_exp();
  void test_log();
  void test_log1p_exp();
  void test_log1p();
  void test_multiply_log();
  void test_pow();
  void test_sin();
  void test_sinh();
  void test_sqrt();
  void test_square();
  void test_tan();
  void test_tanh();
  void test_tgamma();
  void test_trinary_prod_cubes();
  
  void test_scalar_functions() {
    test_acos();
    test_acosh();
    test_asin();
    test_asinh();
    test_atan();
    test_atan2();
    test_atanh();
    test_binary_prod_cubes();
    test_cbrt();
    test_cos();
    test_cosh();
    test_erf();
    test_erfc();
    test_exp();
    test_exp2();
    test_expm1();
    test_fma();
    test_hypot();
    test_inv_cloglog();
    test_inv_logit();
    test_inv_sqrt();
    test_inv_square();
    test_inv();
    test_lgamma();
    test_log_diff_exp();
    test_log_sum_exp();
    test_log();
    test_log1p_exp();
    test_log1p();
    test_multiply_log();
    test_pow();
    test_sin();
    test_sinh();
    test_sqrt();
    test_square();
    test_tan();
    test_tanh();
    test_tgamma();
    test_trinary_prod_cubes();
  }
  
  // acos
  template <typename T>
  class acos_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return acos(v);
      
    }
    static std::string name() { return "acos"; }
  };
  
  void test_acos() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<acos_func>(x);
  }
  
  // acosh
  template <typename T>
  class acosh_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return acosh(v);
      
    }
    static std::string name() { return "acosh"; }
  };
  
  void test_acosh() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<acosh_func>(x);
  }
  
  // asin
  template <typename T>
  class asin_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return asin(v);
      
    }
    static std::string name() { return "asin"; }
  };
  
  void test_asin() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<asin_func>(x);
  }
  
  // asinh
  template <typename T>
  class asinh_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return asinh(v);
      
    }
    static std::string name() { return "asinh"; }
  };
  
  void test_asinh() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<asinh_func>(x);
  }
  
  // atan
  template <typename T>
  class atan_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return atan(v);
      
    }
    static std::string name() { return "atan"; }
  };
  
  void test_atan() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<atan_func>(x);
  }
  
  // atan2
  template <typename T>
  class atan2_vv_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v1 = x[0];
      T v2 = x[1];
      return atan2(v1, v2);
      
    }
    static std::string name() { return "atan2_vv"; }
  };
  
  template <typename T>
  class atan2_vd_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return atan2(v, 0.4847);
      
    }
    static std::string name() { return "atan2_vd"; }
  };
  
  template <typename T>
  class atan2_dv_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return atan2(0.3898, v);
      
    }
    static std::string name() { return "atan2_dv"; }
  };
  
  void test_atan2() {
    Eigen::VectorXd x1 = Eigen::VectorXd::Ones(2);
    x1[0] *= 0.576;
    x1[1] *= -0.294;
    
    tests::test_function<atan2_vv_func>(x1);
    
    Eigen::VectorXd x2 = Eigen::VectorXd::Ones(1);
    x2 *= 0.576;
    
    tests::test_function<atan2_vd_func>(x2);
    tests::test_function<atan2_dv_func>(x2);
  }
  
  // atanh
  template <typename T>
  class atanh_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return atanh(v);
      
    }
    static std::string name() { return "atanh"; }
  };
  
  void test_atanh() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<atanh_func>(x);
  }
 
  // binary_prod_cubes
  template <typename T>
  class binary_prod_cubes_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v1 = x[0];
      T v2 = x[1];
      return binary_prod_cubes(v1, v2);
      
    }
    static std::string name() { return "binary_prod_cubes"; }
  };
  
  void test_binary_prod_cubes() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(2);
    x[0] = 0.576;
    x[1] = 0.832;
    tests::test_function<binary_prod_cubes_func>(x);
  }

  // cbrt
  template <typename T>
  class cbrt_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return cbrt(v);
      
    }
    static std::string name() { return "cbrt"; }
  };
  
  void test_cbrt() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<cbrt_func>(x);
  }
  
  // cos
  template <typename T>
  class cos_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return cos(v);
      
    }
    static std::string name() { return "cos"; }
  };
  
  void test_cos() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<cos_func>(x);
  }
  
  // cosh
  template <typename T>
  class cosh_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return cosh(v);
      
    }
    static std::string name() { return "cosh"; }
  };
  
  void test_cosh() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<cosh_func>(x);
  }

  // erf
  template <typename T>
  class erf_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return erf(v);
      
    }
    static std::string name() { return "erf"; }
  };
  
  void test_erf() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<erf_func>(x);
  }
  
  // erfc
  template <typename T>
  class erfc_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return erfc(v);
      
    }
    static std::string name() { return "exp"; }
  };
  
  void test_erfc() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<erfc_func>(x);
  }
  
  // exp
  template <typename T>
  class exp_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return exp(v);
      
    }
    static std::string name() { return "exp"; }
  };
  
  void test_exp() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<exp_func>(x);
  }
  
  // exp2
  template <typename T>
  class exp2_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return exp2(v);
      
    }
    static std::string name() { return "exp2"; }
  };
  
  void test_exp2() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<exp2_func>(x);
  }
  
  // expm1
  template <typename T>
  class expm1_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return expm1(v);
      
    }
    static std::string name() { return "expm1"; }
  };
  
  void test_expm1() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<expm1_func>(x);
  }
  
  // fma
  template <typename T>
  class fma_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v1 = x[0];
      T v2 = x[1];
      T v3 = x[2];
      return fma(v1, v2, v3);
      
    }
    static std::string name() { return "fma"; }
  };
  
  void test_fma() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(3);
    x[0] *= -2.483;
    x[1] *= 0.576;
    x[2] *= 1.384;
    tests::test_function<fma_func>(x);
  }
  
  // hypot
  template <typename T>
  class hypot_vv_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v1 = x[0];
      T v2 = x[1];
      return hypot(v1, v2);
      
    }
    static std::string name() { return "hypot_vv"; }
  };
  
  template <typename T>
  class hypot_vd_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return hypot(v, 0.4847);
      
    }
    static std::string name() { return "hypot_vd"; }
  };
  
  template <typename T>
  class hypot_dv_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return hypot(0.3898, v);
      
    }
    static std::string name() { return "hypot_dv"; }
  };
  
  void test_hypot() {
    Eigen::VectorXd x1 = Eigen::VectorXd::Ones(2);
    x1[0] *= 0.576;
    x1[1] *= -0.294;
    
    tests::test_function<hypot_vv_func>(x1);
    
    Eigen::VectorXd x2 = Eigen::VectorXd::Ones(1);
    x2 *= 0.576;
    
    tests::test_function<hypot_vd_func>(x2);
    tests::test_function<hypot_dv_func>(x2);
  }

  // inv_cloglog
  template <typename T>
  class inv_cloglog_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return inv_cloglog(v);
      
    }
    static std::string name() { return "inv_cloglog"; }
  };
  
  void test_inv_cloglog() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<inv_cloglog_func>(x);
  }
  
  // inv_logit
  template <typename T>
  class inv_logit_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return inv_logit(v);
      
    }
    static std::string name() { return "inv_logit"; }
  };
  
  void test_inv_logit() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<inv_logit_func>(x);
  }
  
  // inv_sqrt
  template <typename T>
  class inv_sqrt_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return inv_sqrt(v);
      
    }
    static std::string name() { return "inv_sqrt"; }
  };
  
  void test_inv_sqrt() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<inv_sqrt_func>(x);
  }
  
  // inv_square
  template <typename T>
  class inv_square_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return inv_square(v);
      
    }
    static std::string name() { return "inv_square"; }
  };
  
  void test_inv_square() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<inv_square_func>(x);
  }
  
  // inv
  template <typename T>
  class inv_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return inv(v);
      
    }
    static std::string name() { return "inv"; }
  };
  
  void test_inv() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<inv_func>(x);
  }
  
  // lgamma
  template <typename T>
  struct lgamma_func {
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return lgamma(v);
      
    }
    static std::string name() { return "lgamma"; }
  };
  
  void test_lgamma() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 4.584;
    tests::test_function<lgamma_func>(x);
  }
  
  // log_diff_exp
  template <typename T>
  class log_diff_exp_vv_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v1 = x[0];
      T v2 = x[1];
      return log_diff_exp(v1, v2);
      
    }
    static std::string name() { return "log_diff_exp_vv"; }
  };
  
  template <typename T>
  class log_diff_exp_vd_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return log_diff_exp(v, 0.5);
      
    }
    static std::string name() { return "log_diff_exp_vd"; }
  };
  
  template <typename T>
  class log_diff_exp_dv_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return log_diff_exp(0.5, v);
      
    }
    static std::string name() { return "log_diff_exp_dv"; }
  };
  
  void test_log_diff_exp() {
    Eigen::VectorXd x1 = Eigen::VectorXd::Ones(2);
    x1[0] *= 1.0;
    x1[1] *= 0.5;
    
    tests::test_function<log_diff_exp_vv_func>(x1);
    
    x1[0] = -1.0;
    tests::test_function<log_diff_exp_vv_func>(x1);
    
    Eigen::VectorXd x2 = Eigen::VectorXd::Ones(1);
    
    tests::test_function<log_diff_exp_vd_func>(x2);
    tests::test_function<log_diff_exp_dv_func>(x2);
    
    x2[0] = -1.0;
    tests::test_function<log_diff_exp_vd_func>(x2);
    tests::test_function<log_diff_exp_dv_func>(x2);
    
  }
  
  // log_sum_exp
  template <typename T>
  class log_sum_exp_vv_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v1 = x[0];
      T v2 = x[1];
      return log_sum_exp(v1, v2);
      
    }
    static std::string name() { return "log_sum_exp_vv"; }
  };
  
  template <typename T>
  class log_sum_exp_vd_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return log_sum_exp(v, 0.5);
      
    }
    static std::string name() { return "log_sum_exp_vd"; }
  };
  
  template <typename T>
  class log_sum_exp_dv_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return log_sum_exp(0.5, v);
      
    }
    static std::string name() { return "log_sum_exp_dv"; }
  };
  
  void test_log_sum_exp() {
    Eigen::VectorXd x1 = Eigen::VectorXd::Ones(2);
    x1[0] *= 1.0;
    x1[1] *= 0.5;
    
    tests::test_function<log_sum_exp_vv_func>(x1);
    
    x1[0] = -1.0;
    tests::test_function<log_sum_exp_vv_func>(x1);
    
    Eigen::VectorXd x2 = Eigen::VectorXd::Ones(1);
    
    tests::test_function<log_sum_exp_vd_func>(x2);
    tests::test_function<log_sum_exp_dv_func>(x2);
    
    x2[0] = -1.0;
    tests::test_function<log_sum_exp_vd_func>(x2);
    tests::test_function<log_sum_exp_dv_func>(x2);
    
  }

  // log
  template <typename T>
  class log_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return log(v);
      
    }
    static std::string name() { return "log"; }
  };
  
  void test_log() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<log_func>(x);
  }

  // log1p_exp
  template <typename T>
  class log1p_exp_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return log1p_exp(v);
      
    }
    static std::string name() { return "log1p_exp"; }
  };

  void test_log1p_exp() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<log1p_exp_func>(x);
    x *= -1.0;
    tests::test_function<log1p_exp_func>(x);
  }
  
  // log1p
  template <typename T>
  class log1p_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return log1p(v);
      
    }
    static std::string name() { return "log1p"; }
  };
  
  void test_log1p() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<log1p_func>(x);
  }
  
  // log2
  template <typename T>
  class log2_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return log2(v);
      
    }
    static std::string name() { return "log2"; }
  };
  
  void test_log2() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<log2_func>(x);
  }
  
  // log10
  template <typename T>
  class log10_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return log10(v);
      
    }
    static std::string name() { return "log10"; }
  };
  
  void test_log10() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<log10_func>(x);
  }
  
  // multiply_log
  template <typename T>
  class multiply_log_vv_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v1 = x[0];
      T v2 = x[1];
      return multiply_log(v1, v2);
      
    }
    static std::string name() { return "multiply_log_vv"; }
  };
  
  template <typename T>
  class multiply_log_vd_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return multiply_log(v, 0.5);
      
    }
    static std::string name() { return "multiply_log_vd"; }
  };
  
  template <typename T>
  class multiply_log_dv_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return multiply_log(0.5, v);
      
    }
    static std::string name() { return "multiply_log_dv"; }
  };
  
  void test_multiply_log() {
    Eigen::VectorXd x1 = Eigen::VectorXd::Ones(2);
    x1[0] *= 1.0;
    x1[1] *= 0.5;
    
    tests::test_function<multiply_log_vv_func>(x1);
    
    Eigen::VectorXd x2 = Eigen::VectorXd::Ones(1);
    
    tests::test_function<multiply_log_vd_func>(x2);
    tests::test_function<multiply_log_dv_func>(x2);
    
  }
  
  // pow
  template <typename T>
  class pow_vv_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v1 = x[0];
      T v2 = x[1];
      return pow(v1, v2);
      
    }
    static std::string name() { return "pow_vv"; }
  };
  
  template <typename T>
  class pow_vd_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return pow(v, 0.4847);
      
    }
    static std::string name() { return "pow_vd"; }
  };
  
  template <typename T>
  class pow_dv_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return pow(0.3898, v);
      
    }
    static std::string name() { return "pow_dv"; }
  };
  
  void test_pow() {
    Eigen::VectorXd x1 = Eigen::VectorXd::Ones(2);
    x1[0] *= 0.576;
    x1[1] *= -0.294;
    
    tests::test_function<pow_vv_func>(x1);
    
    Eigen::VectorXd x2 = Eigen::VectorXd::Ones(1);
    x2 *= 0.576;
    
    tests::test_function<pow_vd_func>(x2);
    tests::test_function<pow_dv_func>(x2);
  }

  // Phi
  template <typename T>
  class Phi_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return Phi(v);
      
    }
    static std::string name() { return "Phi"; }
  };
  
  void test_Phi() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<Phi_func>(x);
  }
  
  // sin
  template <typename T>
  class sin_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return sin(v);
      
    }
    static std::string name() { return "sin"; }
  };
  
  void test_sin() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<sin_func>(x);
  }
  
  // sinh
  template <typename T>
  class sinh_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return sinh(v);
      
    }
    static std::string name() { return "sinh"; }
  };
  
  void test_sinh() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<sinh_func>(x);
  }
  
  // sqrt
  template <typename T>
  class sqrt_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return sqrt(v);
      
    }
    static std::string name() { return "sqrt"; }
  };
  
  void test_sqrt() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<sqrt_func>(x);
  }
  
  // square
  template <typename T>
  class square_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return square(exp(v));
      
    }
    static std::string name() { return "square"; }
  };
  
  void test_square() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<square_func>(x);
  }
  
  // tan
  template <typename T>
  class tan_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return tan(v);
      
    }
    static std::string name() { return "tan"; }
  };
  
  void test_tan() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<tan_func>(x);
  }
  
  // tanh
  template <typename T>
  class tanh_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return tanh(v);
      
    }
    static std::string name() { return "tanh"; }
  };
  
  void test_tanh() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<tanh_func>(x);
  }
  
  // tgamma
  template <typename T>
  struct tgamma_func {
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return tgamma(v);
      
    }
    static std::string name() { return "tgamma"; }
  };
  
  void test_tgamma() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 1.2483;
    tests::test_function<tgamma_func>(x);
  }
  
  // trinary_prod_cubes
  template <typename T>
  class trinary_prod_cubes_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v1 = x[0];
      T v2 = x[1];
      T v3 = x[2];
      return trinary_prod_cubes(v1, v2, v3);
      
    }
    static std::string name() { return "trinary_prod_cubes"; }
  };
  
  void test_trinary_prod_cubes() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(3);
    x[0] = 0.576;
    x[1] = 0.832;
    x[1] = -1.765;
    tests::test_function<trinary_prod_cubes_func>(x);
  }
  
}

#endif
