#ifndef nomad__src__scalar__functions__smooth_functions__polygamma_hpp
#define nomad__src__scalar__functions__smooth_functions__polygamma_hpp

#include <math.h>
#include <src/scalar/constants.hpp>

namespace nomad {

  double digamma(double x);
  double trigamma(double x);
  double quadrigamma(double x);
  
  // Accuracy of these implementation is lacking
  // for small arguments with small_threshold < x << 1.
  
  double digamma(double x) {
  
    double small_threshold = 1e-4;
    double large_threshold = 8;
    
    // Poles at the negative integers
    if (x <= 0 && std::floor(x) == x)
      return std::numeric_limits<double>::infinity();
    
    // Reflect negative values away from the poles
    if (x <= 0 && std::floor(x) != x)
      return digamma(1 - x) - pi / std::tan(pi * x);
  
    // Small value approximation
    if (x <= small_threshold)
      return - 1.0 / x;
    
    // Use recursion relation to slide up to a large argument
    double psi = 0;
    
    while (x < large_threshold) {
      psi += - 1.0 / x;
      ++x;
    }
    
    // Asymptotic expansion for large arguments
    
    // Expansion coefficients given by scaled second Bernoulli Numbers
    // B0 = 1, B1 = 0.5,
    // B2 = 1/6, B4 = - 1/30, B6 = 1/42, B8 = -1/30
    // B3 = B5 = B7 = 0
    const double C1 = 0.5;
    const double C2 = 1.0 / 12.0;
    const double C4 = - 1.0 / 120.0;
    const double C6 = 1.0 / 252.0;
    const double C8 = - 1.0 / 240.0;
    
    double x_inv = 1.0 / x;
    double x2_inv = x_inv * x_inv;
    
    return psi + std::log(x) - x_inv * C1
           - x2_inv * (C2 + x2_inv * (C4 + x2_inv * (C6 + x2_inv * C8)));
    
  }
  
  double trigamma(double x) {
    
    double small_threshold = 1e-4;
    double large_threshold = 8;

    // Poles at the negative integers
    if (x <= 0 && std::floor(x) == x)
      return std::numeric_limits<double>::infinity();
    
    // Reflect negative values away from the poles
    if (x <= 0 && std::floor(x) != x) {
      double r = pi / std::sin(pi * x);
      return - trigamma(1 - x) + r * r;
    }
      
    // Small value approximation
    if (x <= small_threshold)
      return 1 / (x * x);
    
    // Use recursion relation to slide up to a large argument
    double psi = 0;
    
    while (x < large_threshold) {
      psi += 1.0 / (x * x);
      ++x;
    }
    
    // Asymptotic expansion for large arguments
    
    // Nonzero second Bernoulli Numbers
    // B0 = 1, B1 = 0.5,
    // B2 = 1/6, B4 = - 1/30, B6 = 1/42, B8 = -1/30
    // B3 = B5 = B7 = 0
    const double B1 = 0.5;
    const double B2 = 1.0 / 6.0;
    const double B4 = - 1.0 / 30.0;
    const double B6 = 1.0 / 42.0;
    const double B8 = - 1.0 / 30.0;
    
    double x_inv = 1.0 / x;
    double x2_inv = x_inv * x_inv;
    
    return psi + x_inv * (1 + B1 * x_inv + x2_inv * (B2 + x2_inv * (B4 + x2_inv * (B6 + x2_inv * B8))));
  }
  
  double quadrigamma(double x) {
    
    double small_threshold = 1e-4;
    double large_threshold = 8;
    
    // Poles at the negative integers
    if (x <= 0 && std::floor(x) == x)
      return std::numeric_limits<double>::infinity();
    
    // Reflect negative values away from the poles
    if (x <= 0 && std::floor(x) != x) {
      double r = pi / std::sin(pi * x);
      return quadrigamma(1 - x) - 2 * r * r * r * std::cos(pi * x);
    }
      
    // Small value approximation
    if (x <= small_threshold)
      return - 2.0 / (x * x * x);
    
    // Use recursion relation to slide up to a large argument
    double psi = 0;
    
    while (x < large_threshold) {
      psi += - 2.0 / (x * x * x);
      ++x;
    }
    
    // Asymptotic expansion for large arguments
    
    // Expansion coefficients given by (k + 1) * B_{k},
    // where B_{k} are the second Bernoulli Numbers
    // B0 = 1, B1 = 0.5,
    // B2 = 1/6, B4 = - 1/30, B6 = 1/42, B8 = -1/30
    // B3 = B5 = B7 = 0
    const double C1 = 1;
    const double C2 = 0.5;
    const double C4 = - 1.0 / 6.0;
    const double C6 = 1.0 / 6.0;
    const double C8 = - 3.0 / 10.0;
    
    double x_inv = 1.0 / x;
    double x2_inv = x_inv * x_inv;
    
    return psi - x2_inv * (1 + C1 * x_inv + x2_inv * (C2 + x2_inv * (C4 + x2_inv * (C6 + x2_inv * C8))));
    
  }
  

}

#endif
