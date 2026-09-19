#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "torch/Creation.h"

using mini_torch::c10::ScalarType;
using mini_torch::torch::empty;
using mini_torch::torch::ones;
using mini_torch::torch::zeros;
using mini_torch::torch::full;
using mini_torch::torch::Tensor;

void testCreationEmpty(){
   std::cout << "Running testCreationEmpty..."
      << "\n";

   Tensor tensor = empty({2, 3});

   assert(tensor.defined());

   assert(tensor.dim() == 2);

   assert(tensor.size(0) == 2);
   assert(tensor.size(1) == 3);

   assert(tensor.numel() == 6);

   assert(
      tensor.dtype() ==
      ScalarType::Float32);

   assert(tensor.isContiguous());

   std::cout << "  passed\n";
}

void testCreationZeros(){
   std::cout << "Running testCreationZeros...\n";

   Tensor tensor = zeros({2, 3});

   assert(tensor.defined());

   assert(tensor.numel() == 6);

   const float* data = static_cast<const float*>(tensor.data());

   for(std::int64_t i = 0; i < 6; ++i){
      assert(data[i] == static_cast<float>(0));
   }

   std::cout << "  passed\n";
}

void testCreationOnes(){
   std::cout << "Running testCreationOnes...\n";

   Tensor tensor = ones({2, 3});

   assert(tensor.defined());

   assert(tensor.numel() == 6);

   const float* data = static_cast<const float*>(tensor.data());

   for(std::int64_t i = 0; i < 6; ++i){
      assert(data[i] == static_cast<float>(1));
   }

   std::cout << "  passed\n";
}

void testCreationFull(){
   std::cout << "Running testCreationFull...\n";

   Tensor tensor = full({2, 3}, 7.5);

   assert(tensor.defined());

   assert(tensor.numel() == 6);

   const float* data = static_cast<const float*>(tensor.data());

   for(std::int64_t i = 0; i < 6; ++i){
      assert(
         std::fabs(
               data[i] - 7.5f) <
         1e-6f);
   }

   std::cout << "  passed\n";
}

void testCreationShape()
{
   std::cout
      << "Running testCreationShape...\n";

   auto tensor =
      zeros({2, 3, 4});

   assert(tensor.dim() == 3);

   assert(tensor.size(0) == 2);
   assert(tensor.size(1) == 3);
   assert(tensor.size(2) == 4);

   assert(tensor.numel() == 24);

   assert(tensor.stride(0) == 12);
   assert(tensor.stride(1) == 4);
   assert(tensor.stride(2) == 1);

   std::cout
      << "  passed\n";
}

void testCreationScalar()
{
   std::cout
      << "Running testCreationScalar...\n";

   auto tensor =
      full({}, 42.0);

   assert(tensor.defined());

   assert(tensor.dim() == 0);

   assert(tensor.numel() == 1);

   const float* data =
      static_cast<const float*>(
         tensor.data());

   assert(data[0] == 42.0f);

   std::cout
      << "  passed\n";
}

void testCreationZeroSize()
{
   std::cout
      << "Running testCreationZeroSize...\n";

   auto tensor =
      zeros({2, 0, 3});

   assert(tensor.defined());

   assert(tensor.dim() == 3);

   assert(tensor.size(0) == 2);
   assert(tensor.size(1) == 0);
   assert(tensor.size(2) == 3);

   assert(tensor.numel() == 0);

   std::cout
      << "  passed\n";
}

void testCreationRejectsNegativeDimension()
{
   std::cout
      << "Running "
      << "testCreationRejectsNegativeDimension...\n";

   bool threw = false;

   try {
      auto tensor =
         zeros({2, -1, 3});

      (void)tensor;
   }
   catch (const std::exception&) {
      threw = true;
   }

   assert(threw);

   std::cout
      << "  passed\n";
}

void testCreationIndependentStorage()
{
   std::cout
      << "Running "
      << "testCreationIndependentStorage...\n";

   auto a =
      zeros({2, 2});

   auto b =
      ones({2, 2});

   float* a_data =
      static_cast<float*>(
         a.data());

   const float* b_data =
      static_cast<const float*>(
         b.data());

   a_data[0] = 100.0f;

   assert(a_data[0] == 100.0f);

   assert(b_data[0] == 1.0f);

   std::cout
      << "  passed\n";
}

void runCreationTests()
{
   testCreationEmpty();
   testCreationZeros();
   testCreationOnes();
   testCreationFull();
   testCreationShape();
   testCreationScalar();
   testCreationZeroSize();
   testCreationRejectsNegativeDimension();
   testCreationIndependentStorage();
}