#include <iostream>
#include <memory>

#include "../torch/csrc/api/include/torch/Tensor.h"

using mini_torch::at::TensorImpl;
using mini_torch::c10::ScalarType;
using mini_torch::torch::Tensor;

int main(){
   auto impl =
      std::make_shared<TensorImpl>(
         ScalarType::Float32,
         TensorImpl::SizeVector{2, 3});

   Tensor tensor(impl);

   std::cout
      << tensor.repr()
      << "\n";

   float* data = static_cast<float*>(tensor.data());

   for (int i = 0; i < 6; ++i) {
      data[i] =
         static_cast<float>(i + 1);
   }

   std::cout
      << "numel: "
      << tensor.numel()
      << "\n";

   std::cout
      << "dim: "
      << tensor.dim()
      << "\n";

   std::cout
      << "size(0): "
      << tensor.size(0)
      << "\n";

   std::cout
      << "size(1): "
      << tensor.size(1)
      << "\n";

   std::cout
      << "dtype: "
      << mini_torch::c10::scalarTypeName(
            tensor.dtype())
      << "\n";

   std::cout
      << "device: "
      << tensor.device().str()
      << "\n";

   Tensor view =
      tensor.view({3, 2});

   std::cout
      << "\nView:\n"
      << view.repr()
      << "\n";

   return 0;
}