#include <iostream>

void runStorageTests();
void runTensorImplTests();
void runTensorTests();
void runCreationTests();

int main() {
    try {
        runStorageTests();
        runTensorImplTests();
        runTensorTests();
        runCreationTests();

        std::cout << "\n"
                  << "=================================\n"
                  << "All tests passed!\n"
                  << "=================================\n";

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr
            << "\nTEST FAILURE: "
            << e.what()
            << "\n";

        return 1;
    }
}