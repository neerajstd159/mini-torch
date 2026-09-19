#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "ATen/core/TensorImpl.h"

namespace mini_torch{
   namespace torch{

      class Tensor{
         public:
            using SizeVector = std::vector<std::int64_t>;

            Tensor() = default;

            explicit Tensor(std::shared_ptr<at::TensorImpl> impl)
               : impl_(std::move(impl)) {}

            bool defined() const{
               return impl_ != nullptr && impl_->defined();
            }

            int64_t dim() const{
               checkDefined();

               return impl_->dim();
            }

            int64_t size(int64_t dimension) const {
               checkDefined();
               return impl_->size(dimension);
            }

            int64_t stride(int64_t dimension) const {
               checkDefined();
               return impl_->stride(dimension);
            }

            int64_t numel() const {
               checkDefined();
               return impl_->numel();
            }

            std::size_t nbytes() const {
               checkDefined();
               return impl_->nbytes();
            }

            const SizeVector& sizes() const {
               checkDefined();
               return impl_->sizes();
            }

            const SizeVector& strides() const {
               checkDefined();
               return impl_->strides();
            }

            c10::ScalarType dtype() const {
               checkDefined();
               return impl_->dtype();
            }

            const c10::Device& device() const {
               checkDefined();
               return impl_->device();
            }

            bool isContiguous() const {
               checkDefined();
               return impl_->isContiguous();
            }

            void* data() {
               checkDefined();
               return impl_->data();
            }

            const void* data() const {
               checkDefined();
               return impl_->data();
            }

            std::string repr() const {
               checkDefined();
               return impl_->repr();
            }

            std::shared_ptr<at::TensorImpl> impl() const {
               return impl_;
            }

            Tensor view(const SizeVector& new_sizes) const {
               checkDefined();

               return Tensor(
                     impl_->view(new_sizes));
            }

            Tensor clone() const {
               checkDefined();

               return Tensor(
                     impl_->clone());
            }

         private:
            void checkDefined() const {
               if (!defined()) {
                     throw std::runtime_error(
                        "Tensor is undefined");
               }
            }

         private:
            std::shared_ptr<at::TensorImpl> impl_;
      };
      
   }
}