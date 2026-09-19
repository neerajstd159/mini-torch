#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "../../aten/src/ATen/core/TensorImpl.h"

using mini_torch::at::TensorImpl;
using mini_torch::c10::ScalarType;

void testTensorCreation(){
   std::cout << "Running test tensor creation...\n";

   TensorImpl tensor(ScalarType::Float32, {2, 3});

   assert(tensor.defined());
   assert(tensor.dim() == 2);
   assert(tensor.size(0) == 2);
   assert(tensor.size(1) == 3);
   assert(tensor.stride(0) == 3);
   assert(tensor.stride(1) == 1);
   assert(tensor.numel() == 6);
   assert(tensor.nbytes() == 6 * sizeof(float));
   assert(tensor.isContiguous());
}

void testTensorIndexing(){
   std::cout << "Test tensor indexing running...\n";

   TensorImpl tensor(ScalarType::Float32, {2, 3});

   for (int64_t i = 0; i < 2; ++i) {
      for (int64_t j = 0; j < 3; ++j) {
         tensor.item<float>({i, j}) = static_cast<float>(i * 3 + j);
      }
   }

   assert(tensor.item<float>({0, 0}) == 0.0f);
   assert(tensor.item<float>({0, 1}) == 1.0f);
   assert(tensor.item<float>({0, 2}) == 2.0f);

   assert(tensor.item<float>({1, 0}) == 3.0f);
   assert(tensor.item<float>({1, 1}) == 4.0f);
   assert(tensor.item<float>({1, 2}) == 5.0f);

   std::cout << "  passed\n";
}

void testTensorView() {
    std::cout << "Running testTensorView...\n";

    auto tensor =
        std::make_shared<TensorImpl>(
            ScalarType::Float32,
            TensorImpl::SizeVector{2, 3});

    float* data =
        static_cast<float*>(tensor->data());

    for (int i = 0; i < 6; ++i) {
        data[i] = static_cast<float>(i);
    }

    auto view =
        tensor->view({3, 2});

    assert(view->numel() == 6);
    assert(view->size(0) == 3);
    assert(view->size(1) == 2);

    assert(view->stride(0) == 2);
    assert(view->stride(1) == 1);

    assert(view->item<float>({0, 0}) == 0.0f);
    assert(view->item<float>({0, 1}) == 1.0f);

    assert(view->item<float>({1, 0}) == 2.0f);
    assert(view->item<float>({1, 1}) == 3.0f);

    assert(view->item<float>({2, 0}) == 4.0f);
    assert(view->item<float>({2, 1}) == 5.0f);

    // Most important property:
    // TensorImpl and view share Storage.
    assert(
        view->storage().data() ==
        tensor->storage().data());

    std::cout << "  passed\n";
}

void testTensorClone() {
    std::cout << "Running testTensorClone...\n";

    TensorImpl tensor(
        ScalarType::Float32,
        {2, 2});

    tensor.item<float>({0, 0}) = 1.0f;
    tensor.item<float>({0, 1}) = 2.0f;
    tensor.item<float>({1, 0}) = 3.0f;
    tensor.item<float>({1, 1}) = 4.0f;

    auto clone = tensor.clone();

    assert(clone->numel() == tensor.numel());

    assert(clone->item<float>({0, 0}) == 1.0f);
    assert(clone->item<float>({0, 1}) == 2.0f);
    assert(clone->item<float>({1, 0}) == 3.0f);
    assert(clone->item<float>({1, 1}) == 4.0f);

    // Clone must have independent storage.
    assert(
        clone->storage().data() !=
        tensor.storage().data());

    clone->item<float>({0, 0}) = 100.0f;

    assert(
        tensor.item<float>({0, 0}) == 1.0f);

    assert(
        clone->item<float>({0, 0}) == 100.0f);

    std::cout << "  passed\n";
}

void testTensorStorageSharing() {
    std::cout << "Running testTensorStorageSharing...\n";

    auto tensor =
        std::make_shared<TensorImpl>(
            ScalarType::Float32,
            TensorImpl::SizeVector{4});

    assert(tensor->storage().useCount() == 1);

    auto view = tensor->view({2, 2});

    assert(tensor->storage().useCount() == 2);
    assert(view->storage().useCount() == 2);

    view->item<float>({1, 1}) = 42.0f;

    assert(
        tensor->item<float>({3}) == 42.0f);

    view.reset();

    assert(tensor->storage().useCount() == 1);

    std::cout << "  passed\n";
}