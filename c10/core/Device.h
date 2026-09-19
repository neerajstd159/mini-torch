#pragma once

#include <stdexcept>
#include <string>

namespace mini_torch{
   namespace c10{

      enum class DeviceType{
         CPU,
         CUDA
      };

      class Device{
         public:
            explicit Device(DeviceType type = DeviceType::CPU)
            : type_(type){
               if(type == DeviceType::CUDA){
                  throw std::runtime_error("CUDA is not implemented in mini-torch");
               }
            }

            DeviceType type() const{
               return type_;
            }

            bool isCPU() const{
               return type_ == DeviceType::CPU;
            }

            bool isCUDA() const{
               return type_ == DeviceType::CUDA;
            }

            std::string str() const{
               switch(type_){
                  case DeviceType::CPU:
                     return "CPU";
                  case DeviceType::CUDA:
                     return "CUDA";
               }

               return "Unknown";
            }

         private:
            DeviceType type_;
      };
      
   }
}