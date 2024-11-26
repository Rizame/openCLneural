// C standard includes
#include <stdio.h>
#include <iostream>
#include <vector>
#include "inc/input_parse.h"
// OpenCL includes
#ifdef __APPLE__

#include <OpenCL/cl.h>

#else
#include <CL/cl.h>
#endif

// Include the declaration for load_IDX3
std::vector<std::vector<std::vector<uint8_t>>> load_IDX3(const std::string &filename);

int main() {
    try {
        // Path to your IDX3 file
        std::string filename = "../trainData/train-images.idx3-ubyte";

        // Load all images from the file
        auto images = load_IDX3(filename);

        std::vector<std::vector<int>> flat_images;
        for (const auto &image: images) {
            std::vector<int> flat_image;
            for (const auto &row: image) {
                for (uint8_t pixel: row) {
                    flat_image.push_back(static_cast<int>(pixel));
                }
            }
            flat_images.push_back(flat_image);
        }
        std::cout << "Total number of flattened images stored: " << flat_images.size() << "\n";

        // Optional: Print the dimensions of each flattened image
        if (!flat_images.empty()) {
            std::cout << "Each flattened image contains " << flat_images[0].size() << " pixels.\n";
        }
    }
    catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1; // Return non-zero to indicate failure
    }

    try {
        // Path to the IDX1 file
        std::string filename = "../trainData/train-labels.idx1-ubyte";

        // Load the labels into a dynamically allocated array
        size_t num_labels = 0;
        uint8_t *labels = load_IDX1_to_array(filename, num_labels);

        // Print the total number of labels
        std::cout << "Total number of labels: " << num_labels << "\n";

        // Optional: Print the first 10 labels to verify
        std::cout << "First 10 labels: ";
        for (size_t i = 0; i < 10 && i < num_labels; ++i) {
            std::cout << static_cast<int>(labels[i]) << " ";
        }
        std::cout << "\n";

        // Free the dynamically allocated memory
        delete[] labels;

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1; // Return non-zero to indicate failure
    }

    return 0; // Success
}
