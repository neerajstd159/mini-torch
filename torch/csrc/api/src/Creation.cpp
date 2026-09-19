#include "torch/Creation.h"

#include <cstring>
#include <memory>
#include <stdexcept>

#include "../../../../aten/src/ATen/core/TensorImpl.h"
#include "../../../../c10/core/ScalarType.h"

namespace mini_torch{
   namespace torch{
      namespace{

         constexpr c10::ScalarType kDefaultDtype = c10::ScalarType::Float32;

         at::TensorImpl::SizeVector makeSizeVector(std::initializer_list<std::int64_t> sizes){
            return at::TensorImpl::SizeVector(sizes.begin(), sizes.end());
         }

         Tensor createEmpty(std::initializer_list<std::int64_t> sizes){
            auto size_vector = makeSizeVector(sizes);

            auto impl = std::make_shared<at::TensorImpl>(kDefaultDtype, size_vector);

            return Tensor(std::move(impl));
         }

         void fillFloat32(Tensor& tensor, float value){
            if(!tensor.defined()){
               throw std::runtime_error("Cannot fill an undefined tensor");
            }

            const auto numel = tensor.numel();

            if(numel == 0){
               return;
            }

            float* data = static_cast<float*>(tensor.data());

            for(std::int64_t i = 0; i < numel; ++i){
               data[i] = value;
            }
         }
      }

      Tensor empty(std::initializer_list<std::int64_t> sizes){
         return createEmpty(sizes);
      }

      Tensor zeros(std::initializer_list<std::int64_t> sizes){
         auto tensor = createEmpty(sizes);

         if(tensor.numel() == 0){
            return tensor;
         }

         std::memset(tensor.data(), 0, tensor.nbytes());

         return tensor;
      }

      Tensor ones(std::initializer_list<std::int64_t> sizes){
         auto tensor = createEmpty(sizes);

         fillFloat32(tensor, 1.0f);

         return tensor;
      }

      Tensor full(std::initializer_list<std::int64_t> sizes, double value){
         auto tensor =
            createEmpty(sizes);

         fillFloat32(
            tensor,
            static_cast<float>(value));

         return tensor;
      }
   }
}