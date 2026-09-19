#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "../../../..//c10/core/Device.h"
#include "../../../..//c10/core/ScalarType.h"
#include "Storage.h"

namespace mini_torch{
   namespace at{

      class TensorImpl{
         public:
            using SizeVector = std::vector<std::int64_t>;

            TensorImpl(
               c10::ScalarType dtype,
               const SizeVector& sizes,
               c10::Device device = c10::Device()
            );

            TensorImpl(
               Storage storage,
               c10::ScalarType dtype,
               const SizeVector& sizes,
               const SizeVector& strides,
               std::int64_t storage_offset = 0,
               c10::Device device = c10::Device()
            );

            ~TensorImpl() = default;

            TensorImpl(const TensorImpl&) = default;

            TensorImpl(TensorImpl&&) noexcept = default;

            TensorImpl& operator=(const TensorImpl&) = default;

            TensorImpl& operator=(TensorImpl&&) noexcept = default;


            // ------------------------------------------------------------
            // Metadata
            // ------------------------------------------------------------ 

            c10::ScalarType dtype() const;

            const c10::Device& device() const;

            int64_t dim() const;

            int64_t size(int64_t dimension) const;
            
            int64_t stride(int64_t dimension) const;

            const SizeVector& sizes() const;

            const SizeVector& strides() const;

            int64_t storageOffset() const;

            std::int64_t numel() const;

            std::size_t nbytes() const;

            bool defined() const;

            bool isContiguous() const;

            // ------------------------------------------------------------
            // Storage
            // ------------------------------------------------------------

            Storage& storage();

            const Storage& storage() const;

            void* data();

            const void* data() const;


            // ------------------------------------------------------------
            // Indexing
            // ------------------------------------------------------------
            
            std::size_t storageIndex(const SizeVector& indices) const;

            void* dataAt(const SizeVector& indices);

            const void* dataAt(const SizeVector& indices) const;

            template <typename T>
            T& item(const SizeVector& indices){
               return *static_cast<T*>(dataAt(indices));
            }

            template <typename T>
            const T& item(const SizeVector& indices) const{
               return *static_cast<const T*>(dataAt(indices));
            }


            // ------------------------------------------------------------
            // View / reshape
            // ------------------------------------------------------------

            std::shared_ptr<TensorImpl> view(const SizeVector& new_sizes) const;


            // ------------------------------------------------------------
            // Copy
            // ------------------------------------------------------------

            std::shared_ptr<TensorImpl> clone() const;


            // ------------------------------------------------------------
            // Debug
            // ------------------------------------------------------------

            std::string repr() const;


         private:
            static SizeVector contiguousStrides(
               const SizeVector& sizes);

            static std::int64_t computeNumel(
               const SizeVector& sizes);

            void validateSizes() const;

            void validateStrides() const;

            void validateStorage() const;

            void checkDimension(int64_t dimension) const;

            void checkIndices(const SizeVector& indices) const;

         private:
            Storage storage_;

            c10::ScalarType dtype_;

            c10::Device device_;

            SizeVector sizes_;

            SizeVector strides_;

            std::int64_t storage_offset_;
      };
      
   }
}