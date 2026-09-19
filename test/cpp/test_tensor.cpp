#include <cassert>
#include <iostream>
#include <memory>

#include "../../torch/csrc/api/include/torch/Tensor.h"

using mini_torch::at::TensorImpl;
using mini_torch::c10::ScalarType;
using mini_torch::torch::Tensor;

void testTensorWrapper(){
   std::cout << "Running testtensorwrapper...\n";

   auto impl = std::make_shared<TensorImpl>(ScalarType::Float32, TensorImpl::SizeVector{2,3});

   Tensor tensor(impl);

   assert(tensor.defined());

   assert(tensor.dim() == 2);

   assert(tensor.size(0) == 2);

   assert(tensor.size(1) == 3);

   assert(tensor.stride(0) == 3);

   assert(tensor.stride(1) == 1);

   assert(tensor.numel() == 6);

   assert(tensor.nbytes() == 6 * sizeof(float));

   assert(tensor.isContiguous());

   std::cout << " Passed"
      << "\n";
}

void testTensorWrapperView(){
   std::cout << "Running testTensorWrapperView..."
      << "\n";

   auto impl = std::make_shared<TensorImpl>(ScalarType::Float32, TensorImpl::SizeVector({2,3}));

   Tensor tensor(impl);

   float* data = static_cast<float*>(tensor.data());

   for(int i = 0; i < 6; ++i){
      data[i] = static_cast<float>(i);
   }

   Tensor view = tensor.view({3, 2});

   assert(view.defined());
   
   assert(view.dim() == 2);

   assert(view.size(0) == 3);

   assert(view.size(1) == 2);

   assert(view.stride(0) == 2);

   assert(view.stride(1) == 1);

   assert(view.impl()->storage().data() == tensor.impl()->storage().data());

   std::cout << "  passed\n";
}

void testTensorWrapperClone(){
   std::cout << "Running testTensorWrapperClone...\n";

   auto impl =
        std::make_shared<TensorImpl>(
            ScalarType::Float32,
            TensorImpl::SizeVector{2, 2});

   Tensor tensor(impl);

   tensor.impl()->item<float>({0, 0}) = 1.0f;
   tensor.impl()->item<float>({0, 1}) = 2.0f;
   tensor.impl()->item<float>({1, 0}) = 3.0f;
   tensor.impl()->item<float>({1, 1}) = 4.0f;

   Tensor clone = tensor.clone();

   assert(clone.defined());

   assert(
      clone.impl()->storage().data() !=
      tensor.impl()->storage().data());

   clone.impl()->item<float>({0, 0}) =
      100.0f;

   assert(
      tensor.impl()->item<float>({0, 0}) ==
      1.0f);

   assert(
      clone.impl()->item<float>({0, 0}) ==
      100.0f);

   std::cout << "  passed\n";
}

void testUndefinedTensor(){
   std::cout << "Running testUndefinedTensor...\n";

   Tensor tensor;

   assert(!tensor.defined());

   bool threw = false;

   try{
      tensor.numel();
   }catch(const std::runtime_error&){
      threw = true;
   }

   assert(threw);

   std::cout << "  passed\n";
}

void runTensorTests(){
   testTensorWrapper();
   testTensorWrapperView();
   testTensorWrapperClone();
   testUndefinedTensor();
}