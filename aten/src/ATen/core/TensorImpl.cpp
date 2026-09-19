#include "TensorImpl.h"

#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace mini_torch{
   namespace at{

      TensorImpl::TensorImpl(
         c10::ScalarType dtype,
         const SizeVector& sizes,
         c10::Device device
      )
      :  storage_(),
         dtype_(dtype),
         device_(device),
         sizes_(sizes),
         strides_(contiguousStrides(sizes)),
         storage_offset_(0){
         
         validateSizes();

         const auto elements = numel();
         if(elements < 0){
            throw std::runtime_error("Tensor numel can not be negative");
         }

         const std::size_t bytes = static_cast<std::size_t>(elements) * c10::elementSize(dtype);

         storage_ = Storage(bytes);

         validateStorage();
      }

      TensorImpl::TensorImpl(
         Storage storage,
         c10::ScalarType dtype,
         const SizeVector& sizes,
         const SizeVector& strides,
         std::int64_t storage_offset,
         c10::Device device
      )
      :  storage_(std::move(storage)),
         dtype_(dtype),
         device_(device),
         sizes_(sizes),
         strides_(strides),
         storage_offset_(storage_offset){
         
         validateSizes();
         validateStrides();

         if(storage_offset_ < 0){
            throw std::runtime_error("storage_offset can not be negative");
         }

         validateStorage();
      }

      TensorImpl::SizeVector TensorImpl::contiguousStrides(const SizeVector& sizes){
         SizeVector strides(sizes.size());
         if(sizes.empty()){
            return strides;
         }

         std::int64_t stride = 1;

         for(std::int64_t i = static_cast<int64_t>(sizes.size()) - 1; i >= 0; --i){
            strides[i] = stride;
            stride *= sizes[i];
         }

         return strides;
      }

      std::int64_t TensorImpl::computeNumel(const SizeVector& sizes){
         if (sizes.empty()) {
            return 1;
         }

         std::int64_t result = 1;

         for(auto size: sizes){
            if(size < 0){
               throw std::runtime_error("Tensor dimension cannot be negative");
            }

            if(size == 0){
               return 0;
            }

            if(result > std::numeric_limits<std::int64_t>::max() / size){
               throw std::overflow_error("Tensor numel overflow");
            }

            result *= size;
         }

         return result;
      }

      void TensorImpl::validateSizes() const{
         for(auto size: sizes_){
            if(size < 0){
               throw std::runtime_error("Tensor dimension must be >= 0");
            }
         }
      }

      void TensorImpl::validateStrides() const{
         if(sizes_.size() != strides_.size()){
            throw std::runtime_error("sizes and strides must have the same length");
         }

         for(auto stride: strides_){
            if(stride < 0){
               throw std::runtime_error("Negative strides are not supported");
            }
         }
      }

      void TensorImpl::validateStorage() const{
         if(!storage_.defined()){
            throw std::runtime_error("TensorImpl requires defined storage");
         }

         if(numel() == 0){
            return;
         }

         std::int64_t max_offset = storage_offset_;

         for(std::size_t i = 0; i < sizes_.size(); ++i){
            if (sizes_[i] == 0) {
               return;
            }

            max_offset += (sizes_[i] - 1) * strides_[i];
         }

         const auto required_elements = max_offset + 1;

         const auto required_bytes = static_cast<std::size_t>(required_elements) * c10::elementSize(dtype_);

         if(required_bytes > storage_.nbytes()){
            throw std::runtime_error("TensorImpl exceeds underlying Storage");
         }
      }

      void TensorImpl::checkDimension(int64_t dimension) const {

         if (dimension < 0 ||
            dimension >= dim()) {

            throw std::out_of_range("Tensor dimension out of range");
         }
      }
      
      void TensorImpl::checkIndices(const SizeVector& indices) const{
         if(indices.size() != sizes_.size()){
            throw std::invalid_argument("Number of indices must equal to tensor dimension");
         }

         for(std::size_t i = 0; i < indices.size(); ++i){
            if(indices[i] < 0 || indices[i] >= sizes_[i]){
               throw std::out_of_range("Tensor index out of range");
            }
         }
      }

      c10::ScalarType TensorImpl::dtype() const {
         return dtype_;
      }

      const c10::Device& TensorImpl::device() const {
         return device_;
      }

      int64_t TensorImpl::dim() const {
         return static_cast<int64_t>(sizes_.size());
      }

      int64_t TensorImpl::size(int64_t dimension) const {

         checkDimension(dimension);

         return sizes_[dimension];
      }

      int64_t TensorImpl::stride(int64_t dimension) const {

         checkDimension(dimension);

         return strides_[dimension];
      }

      const TensorImpl::SizeVector& TensorImpl::sizes() const {
         return sizes_;
      }

      const TensorImpl::SizeVector& TensorImpl::strides() const {
         return strides_;
      }

      int64_t TensorImpl::storageOffset() const {
         return storage_offset_;
      }

      std::int64_t TensorImpl::numel() const {
         return computeNumel(sizes_);
      }

      std::size_t TensorImpl::nbytes() const{
         return static_cast<std::size_t>(numel()) * c10::elementSize(dtype_);
      }

      bool TensorImpl::defined() const {
         return storage_.defined();
      }

      bool TensorImpl::isContiguous() const{
         if(sizes_.empty()){
            return true;
         }

         std::int64_t expected_stride = 1;

         for(std::int64_t i = static_cast<std::int64_t>(sizes_.size()) - 1; i >= 0; --i){
            if(sizes_[i] == 0){
               return true;
            }

            if(strides_[i] != expected_stride){
               return false;
            }

            expected_stride *= sizes_[i];
         }

         return true;
      }

      Storage& TensorImpl::storage() {
         return storage_;
      }

      const Storage& TensorImpl::storage() const {
         return storage_;
      }

      void* TensorImpl::data(){
         if(!storage_.defined()){
            return nullptr;
         }

         auto* base = static_cast<std::byte*>(storage_.data());

         return base + storage_offset_ * c10::elementSize(dtype_);
      }

      const void* TensorImpl::data() const {
         if (!storage_.defined()) {
            return nullptr;
         }

         const auto* base =
            static_cast<const std::byte*>(storage_.data());

         return base +
            storage_offset_ *
            c10::elementSize(dtype_);
      }

      std::size_t TensorImpl::storageIndex(const SizeVector& indices) const{
         checkIndices(indices);

         std::size_t offset = storage_offset_;

         for(std::size_t i = 0; i < indices.size(); ++i){
            offset += indices[i] * strides_[i];
         }

         return static_cast<std::size_t>(offset);
      }

      void* TensorImpl::dataAt(const SizeVector& indices){
         const auto index = storageIndex(indices);

         auto* base = static_cast<std::byte*>(storage_.data());

         return base + index * c10::elementSize(dtype_);
      }

      const void* TensorImpl::dataAt(const SizeVector& indices) const {

         const auto index =
            storageIndex(indices);

         const auto* base =
            static_cast<const std::byte*>(storage_.data());

         return base +
            index *
            c10::elementSize(dtype_);
      }

      std::shared_ptr<TensorImpl> TensorImpl::view(const SizeVector& new_sizes) const{
         if (!isContiguous()) {
            throw std::runtime_error(
                  "view() currently requires contiguous TensorImpl");
         }

         if (computeNumel(new_sizes) != numel()) {
            throw std::runtime_error(
                  "view() cannot change number of elements");
         }

         return std::make_shared<TensorImpl>(
            storage_,
            dtype_,
            new_sizes,
            contiguousStrides(new_sizes),
            storage_offset_,
            device_);
      }
      
      std::shared_ptr<TensorImpl> TensorImpl::clone() const{
         auto result = std::make_shared<TensorImpl>(dtype_, sizes_, device_);

         if(numel() > 0){
            std::memcpy(result->data(), data(), nbytes());
         }

         return result;
      }

      std::string TensorImpl::repr() const {

         std::ostringstream out;

         out << "TensorImpl("
            << "dtype="
            << c10::scalarTypeName(dtype_)
            << ", device="
            << device_.str()
            << ", sizes=[";

         for (std::size_t i = 0;
               i < sizes_.size();
               ++i) {

            if (i > 0) {
                  out << ", ";
            }

            out << sizes_[i];
         }

         out << "]"
            << ", strides=[";

         for (std::size_t i = 0;
               i < strides_.size();
               ++i) {

            if (i > 0) {
                  out << ", ";
            }

            out << strides_[i];
         }

         out << "]"
            << ", storage_offset="
            << storage_offset_
            << ", numel="
            << numel()
            << ", contiguous="
            << std::boolalpha
            << isContiguous()
            << ")";

         return out.str();
      }

   }
}