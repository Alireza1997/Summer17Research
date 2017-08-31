// example.h
#ifndef KERNEL_EXAMPLE_H_
#define KERNEL_EXAMPLE_H_

template <typename Device, typename T>
struct ExampleFunctor {
  void operator()(const Device& d, int N, const T* input, T* output, int range, int precision);
};

#endif KERNEL_EXAMPLE_H_