#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace mini_torch{
   namespace c10{

      enum class ScalarType{
         Byte,
         Int32,
         Int64,
         Float32,
         Float64
      };

      inline std::size_t elementSize(ScalarType type){
         switch(type){
            case ScalarType::Byte:
               return sizeof(std::uint8_t);
            case ScalarType::Int32:
               return sizeof(std::int32_t);
            case ScalarType::Int64:
               return sizeof(std::int64_t);
            case ScalarType::Float32:
               return sizeof(float);
            case ScalarType::Float64:
               return sizeof(double);
         }

         throw std::runtime_error("Unknown ScalarType");
      }

      inline const char* scalarTypeName(ScalarType type){
         switch(type){
            case ScalarType::Byte:
               return "Byte";
            case ScalarType::Int32:
               return "Int32";
            case ScalarType::Int64:
               return "Int64";
            case ScalarType::Float32:
               return "Float32";
            case ScalarType::Float64:
               return "Float64";
         }

         return "Unknown";
      }

   }
}