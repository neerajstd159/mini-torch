#include "Storage.h"

#include <algorithm>
#include <cstring>

namespace mini_torch{
   namespace at{

      Storage::Data::Data(std::size_t size)
         : data(nullptr), size(size){
            if(size > 0){
               data = std::make_unique<std::byte[]>(size);
            }
         }

      
      Storage::Storage()
         : data_(nullptr){}


      Storage::Storage(std::size_t nbytes)
         : data_(std::make_shared<Data>(nbytes)){}


      void* Storage::data(){
         if(!data_){
            return nullptr;
         }

         return data_->data.get();
      }


      const void* Storage::data() const {
         if (!data_) {
            return nullptr;
         }

         return data_->data.get();
      }


      std::size_t Storage::nbytes() const{
         if(!data_){
            return 0;
         }

         return data_->size;
      }


      bool Storage::defined() const{
         return static_cast<bool>(data_);
      }


      long Storage::useCount() const{
         if(!data_){
            return 0;
         }

         return data_.use_count();
      }


      void Storage::resize(std::size_t new_size){
         if(!data_){
            data_ = std::make_shared<Data>(new_size);
            return;
         }

         if(new_size == data_->size){
            return;
         }

         auto new_data = std::make_shared<Data>(new_size);

         const std::size_t bytes_to_copy = std::min(data_->size, new_size);

         if(bytes_to_copy > 0){
            std::memcpy(new_data->data.get(), data_->data.get(), bytes_to_copy);
         }

         data_ = std::move(new_data);
      }
      
   }
}