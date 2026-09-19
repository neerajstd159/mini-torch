#pragma once

#include <cstdint>
#include <initializer_list>

#include "torch/Tensor.h"

namespace mini_torch{
   namespace torch{

      Tensor empty(std::initializer_list<int64_t> sizes);

      Tensor zeros(std::initializer_list<int64_t> sizes);

      Tensor ones(std::initializer_list<int64_t> sizes);

      Tensor full(std::initializer_list<int64_t> sizes, double value);
      
   }
}