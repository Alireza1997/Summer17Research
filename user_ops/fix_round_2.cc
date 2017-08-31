// example.cc
#define EIGEN_USE_THREADS
#include "fix_round_2.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// CPU specialization of actual computation.
template <typename T>
struct ExampleFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, int N, const T* input, T* output, int range, int precision) {
    for (int i = 0; i < N; i++){
      //std::cout<<input[i]<< "\n";
      if(input[i] >  (range)){

        output[i] =  (range);        
      }else if(input[i] <  (-range)){

        output[i] =  (-range);
      }else{

        output[i] = input[i]* precision;
        output[i] =  ((int)(output[i]+ 0.5));
        output[i] = output[i]/ precision;
      }
      //std::cout<<output[i] <<"\n";
    } 
  }
};


REGISTER_OP("FixRound")
    .Attr("T: {float, int32}")
    .Input("to_round: T")
    .Input("definition: T")
    .Output("roundeded: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class FixRoundOp : public OpKernel {
 public:
  explicit FixRoundOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    const Tensor& ILFL = context->input(1);

    auto range_precision = ILFL.flat<float>();

    //auto input = input_tensor.flat<float>();

    int range = pow(2,(int)range_precision(0));
    int precision = pow(2,(int)range_precision(1));

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    //auto output = output_tensor->flat<float>();

    // round all elements to fix point.
    //const int N = input.size();
    ExampleFunctor<Device, T>()(
        context->eigen_device<Device>(),
        static_cast<int>(input_tensor.NumElements()),
        input_tensor.flat<T>().data(),
        output_tensor->flat<T>().data(),
        range,
        precision);
        
  }
};


REGISTER_OP("FixRoundGrad")
    .Attr("T: {float, int32}")
    .Input("grad: T")
    .Input("to_round: T")
    .Input("def: T")
    .Output("to_round_grad: T")
    .Output("definition_grad: T");

template <typename Device, typename T>
class FixRoundGradOp : public OpKernel {
 public:
  explicit FixRoundGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& gradient = context->input(0);
    const Tensor& ILFL = context->input(2);

    auto range_precision = ILFL.flat<float>();
    //auto input = gradient.flat<float>();

    //std::cout << range_precision(0) << " " << range_precision(1) <<std::endl;
    int range = pow(2,(int)range_precision(0));
    int precision = pow(2,(int)range_precision(1));
    //std::cout << "rp: " << range << " " << precision <<std::endl;

    // Create an output tensor
    Tensor* to_round_grad = NULL;
    Tensor* definition_grad = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, gradient.shape(),
                                                     &to_round_grad));

    OP_REQUIRES_OK(context, context->allocate_output(1, ILFL.shape(),
                                                     &definition_grad));
    //auto output = to_round_grad->flat<float>();

    // round all elements to fix point.
    ExampleFunctor<Device, T>()(
        context->eigen_device<Device>(),
        static_cast<int>(gradient.NumElements()),
        gradient.flat<T>().data(),
        to_round_grad->flat<T>().data(),
        range,
        precision);
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("FixRound").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      FixRoundOp<CPUDevice, T>);                                \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("FixRoundGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      FixRoundGradOp<CPUDevice, T>);
REGISTER_CPU(float);
//REGISTER_CPU(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("FixRound").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      FixRoundOp<GPUDevice, T>);                                  \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("FixRoundGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      FixRoundGradOp<GPUDevice, T>);
REGISTER_GPU(float);
//REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA