#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>

namespace mini_torch{
   namespace at{

      class Storage{
         public:
            Storage();

            explicit Storage(std::size_t nbytes);

            Storage(const Storage&) = default;

            Storage(Storage&&) noexcept = default;

            Storage& operator=(const Storage&) = default;

            Storage& operator=(Storage&&) noexcept = default;

            ~Storage() = default;

            void* data();

            const void* data() const;

            std::size_t nbytes() const;

            bool defined() const;

            long useCount() const;

            void resize(std::size_t new_size);
            
         private:
            struct Data{
               std::unique_ptr<std::byte[]> data;
               std::size_t size;

               explicit Data(std::size_t size);
            };

            std::shared_ptr<Data> data_;
      };

   }
}