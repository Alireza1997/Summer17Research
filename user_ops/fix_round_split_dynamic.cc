#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <math.h>

using namespace tensorflow;

REGISTER_OP("FixRoundSplit")
    .Input("to_round: float")
    .Input("fdefinition: float")
    .Input("bdefinition: float")
    .Input("foverflow: float")
    .Input("boverflow: float")
    .Output("roundeded: float")    
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });


class FixRoundSplitOp : public OpKernel {
 public:
  explicit FixRoundSplitOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor, ILFL distribution, overflow counter
    const Tensor& input_tensor = context->input(0);
    const Tensor& ILFL = context->input(1);
    Tensor& foverflow = const_cast<Tensor&>(context->input(3));

    auto range_precision = ILFL.flat<float>();

    auto input = input_tensor.flat<float>();

    auto overflow = foverflow.flat<float>();

    int range = pow(2,(int)range_precision(0));
    int precision = pow(2,(int)range_precision(1));

    //std::cout << range << " " << precision <<std::endl;
    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));

    auto output = output_tensor->flat<float>();
    

    // round all elements to fix point.
    const int N = input.size();
    int of0 = 0;
    for (int i = 0; i < N; i++){
      if(input(i) >  (range)){

        output(i) =  (range);
        of0++;
      }else if(input(i) <  (-range)){

        output(i) =  (-range);
        of0++;
      }else{

        output(i) = input(i)* precision;
        output(i) =  ((int)(output(i)+ 0.5));
        output(i) = output(i)/ precision;

      }
    }

    overflow(0) += (float)of0/N;    
  }
};

REGISTER_OP("FixRoundSplitGrad")
  	.Input("grad: float32")
  	.Input("to_round: float32")
	  .Input("fdefinition: float")
	  .Input("bdefinition: float")
    .Input("foverflow: float")
    .Input("boverflow: float")
  	.Output("to_round_grad: float32")
  	.Output("fdefinition_grad: float32")
  	.Output("bdefinition_grad: float32")
    .Output("foverflowg: float32")
    .Output("boverflowg: float32");

class FixRoundSplitGradOp : public OpKernel {
 public:
  explicit FixRoundSplitGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
  	// Grab the input tensor
    const Tensor& gradient = context->input(0);
    const Tensor& ILFL = context->input(3);
    Tensor& boverflow = const_cast<Tensor&>(context->input(5));

    auto range_precision = ILFL.flat<float>();
    auto input = gradient.flat<float>();
    auto overflow = boverflow.flat<float>();

    int range = pow(2,(int)range_precision(0));
    int precision = pow(2,(int)range_precision(1));

    // Create an output tensor
    Tensor* to_round_grad = NULL;
    Tensor* definition_grad = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, gradient.shape(),
                                                     &to_round_grad));

    OP_REQUIRES_OK(context, context->allocate_output(1, ILFL.shape(),
                                                     &definition_grad));

    OP_REQUIRES_OK(context, context->allocate_output(2, ILFL.shape(),
                                                     &definition_grad));
    OP_REQUIRES_OK(context, context->allocate_output(3, ILFL.shape(),
                                                     &definition_grad));
    OP_REQUIRES_OK(context, context->allocate_output(4, ILFL.shape(),
                                                     &definition_grad));
    auto output = to_round_grad->flat<float>();

    // round all elements to fix point.
    const int N = input.size();
    int of0 = 0;
    for (int i = 0; i < N; i++){
      if(input(i) >  (range)){

        output(i) =  (range);
        of0++;
      }else if(input(i) <  (-range)){

        output(i) =  (-range);
        of0++;
      }else{

        output(i) = input(i)* precision;
        output(i) =  ((int)(output(i)+ 0.5));
        output(i) = output(i)/ precision;

      }
    }
    overflow(0) += (float)of0/N;
  }
};

REGISTER_KERNEL_BUILDER(Name("FixRoundSplit").Device(DEVICE_CPU), FixRoundSplitOp);

REGISTER_KERNEL_BUILDER(Name("FixRoundSplitGrad").Device(DEVICE_CPU), FixRoundSplitGradOp);

