#include <iostream>

void testStorage();
void testTensorCreation();
void testTensorIndexing();
void testTensorView();
void testTensorClone();
void testTensorStorageSharing();

int main() {
    try {
        testStorage();
        testTensorCreation();
        testTensorIndexing();
        testTensorView();
        testTensorClone();
        testTensorStorageSharing();

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