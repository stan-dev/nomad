#ifndef nomad__src__meta__expression_templates_hpp
#define nomad__src__meta__expression_templates_hpp

namespace nomad {
  
  /*
  template <typename T>
  class vector_wrapper {
    vector_wrapper(std::string name_in, T& x_in): name(name_in), x(x_in) {};
    std::string name;
    T& x;
  };

  template <typename T, bool is_array, bool throw_if_accessed>
  class VectorView<const T, is_array, throw_if_accessed> {
  public:
    typedef typename scalar_type<T>::type scalar_t;
    
    VectorView(const scalar_t& c) : x_(&c) { }
    
    VectorView(const scalar_t* x) : x_(x) { }
    
    VectorView(const std::vector<scalar_t>& v) : x_(&v[0]) { }
    
    template <int R, int C>
    VectorView(const Eigen::Matrix<scalar_t, R, C>& m) : x_(&m(0)) { }
    
    const scalar_t& operator[](int i) const {
      if (throw_if_accessed)
        throw std::out_of_range("VectorView: this cannot be accessed");
      if (is_array)
        return x_[i];
      else
        return x_[0];
    }
  private:
    const scalar_t* x_;
  };
  */
  
  // simplify to hold value in common case where it's more efficient
  // Possible to enable if on is_arithmetic here?
  //template <typename T>
  //class VectorView<typename const std::enable_if<std::is_arithmetic<T>::value>::type, false, false> {
  //public:
  //  VectorView(T x) : x_(x) { }
  //  double operator[](int /* i */)  const {
  //    return x_;
  //  }
  //private:
  //  const T x_;
  //};
  
}

#endif
