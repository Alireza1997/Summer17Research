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
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    const Tensor& ILFL = context->input(1);
    Tensor& foverflow = const_cast<Tensor&>(context->input(3)); //have to preform a const_cast to be able to pass by reference

    auto range_precision = ILFL.flat<float>();

    auto input = input_tensor.flat<float>();

    auto overflow = foverflow.flat<float>();

    int range = pow(2,(int)range_precision(0));
    int precision = pow(2,(int)range_precision(1));

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));

    auto output = output_tensor->flat<float>();
    

    // round all elements to fix point.
    const int N = input.size();
    int overflow_count = 0;
    int underflow_count = 0;
    double sum,avg,max = 0;
    double min = 1;
    double tmp;

    // loops through the tensor and applies fixedpoint rounding based on the given range and precision
    // counts overflows and underflows during the process, also saves percent change in rounding
    for (int i = 0; i < N; i++){
      if(input(i) >  (range)){

        output(i) =  (range);
        overflow_count++;
      }else if(input(i) <  (-range)){

        output(i) =  (-range);
        overflow_count++;
      }else{

        output(i) = input(i)* precision;
        output(i) =  ((int)(output(i)+ 0.5));
        output(i) = output(i)/ precision;

        if (output(i) != input(i)){
          underflow_count++;
          tmp = fabs((output(i) - input(i))/input(i));
          sum += tmp;
          if (max < tmp)
            max = tmp;
          if (min > tmp)
            min = tmp;
        }
      }
    }

    avg = sum/N;
    overflow(0) += (float)overflow_count/N; //saves the percentage of elements that overflowed
    overflow(1) = avg;   //saves average percent change during rounding
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
                                                     &to_round_grad)); //the gradient is just the given gradient rounded

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
    int overflow_count = 0;
    int underflow_count = 0;
    double sum,avg,max = 0;
    double min = 1;
    double tmp;

    for (int i = 0; i < N; i++){
      if(input(i) >  (range)){

        output(i) =  (range);
        overflow_count++;
      }else if(input(i) <  (-range)){

        output(i) =  (-range);
        overflow_count++;
      }else{

        output(i) = input(i)* precision;
        output(i) =  ((int)(output(i)+ 0.5));
        output(i) = output(i)/ precision;

        if (output(i) != input(i)){
          underflow_count++;
          tmp = fabs((output(i) - input(i))/input(i));
          sum += tmp;
          if (max < tmp)
            max = tmp;
          if (min > tmp)
            min = tmp;
        }
      }
    }

    avg = sum/N;
    overflow(0) += (float)overflow_count/N;
    overflow(1) = avg;
  }
};

REGISTER_KERNEL_BUILDER(Name("FixRoundSplit").Device(DEVICE_CPU), FixRoundSplitOp);

REGISTER_KERNEL_BUILDER(Name("FixRoundSplitGrad").Device(DEVICE_CPU), FixRoundSplitGradOp);

