#include "../inc/input_parse.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>

uint32_t swap_bytes(uint32_t value) {
    return ((value >> 24) & 0x000000FF) |
           ((value >> 8) & 0x0000FF00) |
           ((value << 8) & 0x00FF0000) |
           ((value << 24) & 0xFF000000);
}

int32_t read_int(std::ifstream &file) {
    uint8_t bytes[4];
    file.read(reinterpret_cast<char*>(bytes), 4);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

std::vector<int> load_IDX1_to_array(const std::string &filename, size_t num_labels) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    // Read the header
    uint32_t magic_number = 0;
    file.read(reinterpret_cast<char *>(&magic_number), 4);

    file.read(reinterpret_cast<char *>(&num_labels), 4);

    // Convert from big-endian to host-endian
    magic_number = swap_bytes(magic_number);
    num_labels = swap_bytes(num_labels);

    if (magic_number != 0x00000801) {
        throw std::runtime_error("Invalid magic number in IDX1 file");
    }

    // Allocate memory for the labels as std::vector<int>
    std::vector<int> labels(num_labels);

    // Read the labels into the vector and cast to int
    std::vector<uint8_t> temp_labels(num_labels);  // Temporary storage as uint8_t
    file.read(reinterpret_cast<char *>(temp_labels.data()), num_labels);

    if (!file) {
        throw std::runtime_error("Error reading labels from file");
    }

    // Convert the labels to int and store them in the vector
    for (size_t i = 0; i < num_labels; ++i) {
        labels[i] = static_cast<int>(temp_labels[i]);
    }

    return labels;
}



std::vector<std::vector<int>> load_IDX3(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    // Read the header
    int32_t magic_number = read_int(file);
    int32_t num_images = read_int(file);
    int32_t num_rows = read_int(file);
    int32_t num_cols = read_int(file);

    if (magic_number != 0x00000803) {
        throw std::runtime_error("Invalid magic number: " + std::to_string(magic_number));
    }

    std::cout << "Magic Number: " << magic_number << "\n";
    std::cout << "Number of Images: " << num_images << "\n";
    std::cout << "Image Size: " << num_rows << "x" << num_cols << "\n";

    // Allocate storage for images as ints
    std::vector<std::vector<int>> images(num_images, std::vector<int>(num_rows * num_cols));

    // Read image data and convert uint8_t to int
    for (int i = 0; i < num_images; ++i) {
        for (int r = 0; r < num_rows; ++r) {
            // Read row data as uint8_t, then cast to int for storage
            std::vector<uint8_t> row(num_cols);
            file.read(reinterpret_cast<char*>(row.data()), num_cols);
            for (int c = 0; c < num_cols; ++c) {
                images[i][r * num_cols + c] = static_cast<int>(row[c]);
            }
        }
    }

    file.close();
    return images;
}