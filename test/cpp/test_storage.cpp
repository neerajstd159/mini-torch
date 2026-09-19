#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>

#include "../../aten/src/ATen/core/Storage.h"

using mini_torch::at::Storage;

void testStorage(){
   std::cout << "Running Storage test...\n";

   Storage storage(16);

   assert(storage.defined());
   assert(storage.nbytes() == 16);
   assert(storage.data() != nullptr);
   assert(storage.useCount() == 1);

   auto copy = storage;

   assert(copy.data() == storage.data());
   assert(storage.useCount() == 2);
   assert(copy.useCount() == 2);

   auto* ptr = static_cast<uint8_t*>(storage.data());

   for(int i = 0; i < 16; ++i){
      ptr[i] = static_cast<uint8_t>(i);
   }

   auto* copy_ptr = static_cast<uint8_t*>(copy.data());
   for (int i = 0; i < 16; ++i) {
      assert(copy_ptr[i] == static_cast<std::uint8_t>(i));
   }

   std::cout << "  passed\n";
}

void runStorageTests(){
   testStorage();
}