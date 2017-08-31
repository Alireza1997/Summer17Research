#ifdef GOOGLE_CUDA

#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS

#include "fix_round_2.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

#define EIGEN_USE_GPU

// Define the CUDA kernel.
template <typename T>
__global__ void ExampleCudaKernel(const int size, const T* in, T* out, int range, int precision) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    out[i] = (in[i] > 32)? 32: in[i];
    out[i] = (in[i] < -32)? 32: in[i];
    out[i] = in[i]* precision;
    out[i] =  ((int)(out[i]+ 0.5));
    out[i] = out[i]/ precision; 
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct ExampleFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, int N, const T* input, T* output, int range, int precision) {
    // for (int i = 0; i < N; i++){
    //   //std::cout<<input[i]<< "\n";
    //   if(input[i] >  (range)){

    //     output[i] =  (range);        
    //   }else if(input[i] <  (-range)){

    //     output[i] =  (-range);
    //   }else{

    //     output[i] = input[i]* precision;
    //     output[i] =  ((int)(output[i]+ 0.5));
    //     output[i] = output[i]/ precision;
    //   }
    //   //std::cout<<output[i] <<"\n";
    // }
  // void operator()(const GPUDevice& d, int size, const T* in, T* out) {
    CudaLaunchConfig config = GetCudaLaunchConfig(N, d);

    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.
    // int block_count = 1024;
    // int thread_per_block = 20;
    ExampleCudaKernel<T>
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(N, input, output, range, precision);
  }
};

// Instantiate functors for the types of OpKernels registered.
typedef Eigen::GpuDevice GPUDevice;
template struct ExampleFunctor<GPUDevice, float>;
template struct ExampleFunctor<GPUDevice, int32>;

#endif  // GOOGLE_CUDA