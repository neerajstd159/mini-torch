#include <iostream>
#include <memory>
#include <vector>

#include "../aten/src/ATen/core/TensorImpl.h"

using mini_torch::at::TensorImpl;
using mini_torch::c10::ScalarType;

int main(){
   auto tensor = std::make_shared<TensorImpl>(ScalarType::Float32, std::vector<int64_t>{2, 3});

   std::cout << tensor->repr() << "\n";

   float* data = static_cast<float*>(tensor->data());

   for(int i = 0; i < 6; ++i){
      data[i] = static_cast<float>(i+1);
   }

   std::cout << "\n Tensor values :\n";

   for(int64_t i = 0; i < 2; ++i){
      for(int64_t j = 0; j < 3; ++j){
         std::cout << tensor->item<float>({i, j});
      }

      std::cout << "\n";
   }

   std::cout << "\n storage bytes :"
            << tensor->storage().nbytes()
            << "\n";
   

   std::cout << "Storage references: "
      << tensor->storage().useCount()
      << "\n";
      
   auto view = tensor->view({3,2});

   std::cout << "\n View: \n"
      << view->repr()
      << "\n";


   std::cout << "View values: \n";
   for(int64_t i = 0; i < 3; ++i){
      for(int64_t j = 0; j < 2; ++j){
         std::cout << view->item<float>({i, j})
            << " ";
      }

      std::cout << "\n";
   }

   std::cout << "\nStorage references after view: "
            << tensor->storage().useCount()
            << "\n";

   auto clone = tensor->clone();

   std::cout << "\nClone:\n";
   std::cout << clone->repr() << "\n";

   return 0;
}