#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

template <typename Device, typename T>
class FixRoundOp : public OpKernel {
 public:
  explicit FixRoundOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<T>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->flat<T>();

    // round all elements to fix point.
    const int N = input.size();
    for (int i = 0; i < N; i++){
      if(output(i) > (T)(32)){

        output(i) = (T)(32);
      }else if(output(i) < (T)(-32)){

        output(i) = (T)(-32);
      }else{

        output(i) = output(i)*(T)32768;
        output(i) = (T)((int)(output(i)+(T)0.5));
        output(i) = output(i)/(T)32768;
      }
    }     
  }
};

REGISTER_KERNEL_BUILDER(Name("FixRound").Device(DEVICE_CPU), FixRoundOp);
