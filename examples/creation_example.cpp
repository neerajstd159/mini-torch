#include <iostream>

#include "torch/Creation.h"

using mini_torch::torch::empty;
using mini_torch::torch::full;
using mini_torch::torch::ones;
using mini_torch::torch::zeros;

void printFloatTensor(
    const char* name,
    const mini_torch::torch::Tensor& tensor)
{
   std::cout
      << name
      << ":\n";

   const float* data =
      static_cast<const float*>(
         tensor.data());

   for (std::int64_t i = 0;
      i < tensor.numel();
      ++i) {

      std::cout
         << data[i]
         << " ";
   }

   std::cout
      << "\n\n";
}

int main()
{
   auto empty_tensor =
      empty({2, 3});

   auto zero_tensor =
      zeros({2, 3});

   auto one_tensor =
      ones({2, 3});

   auto full_tensor =
      full({2, 3}, 7.5);

   std::cout
      << "Empty tensor:\n"
      << empty_tensor.repr()
      << "\n\n";

   printFloatTensor(
      "Zeros",
      zero_tensor);

   printFloatTensor(
      "Ones",
      one_tensor);

   printFloatTensor(
      "Full",
      full_tensor);

   return 0;
}