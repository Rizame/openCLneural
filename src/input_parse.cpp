//
// Created by Anton Bucataru on 25.11.2024.
//
#include "../inc/input_parse.h"

int32_t read_int(std::ifstream &file) {
    uint8_t bytes[4];
    file.read(reinterpret_cast<char*>(bytes), 4);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

std::vector<std::vector<std::vector<uint8_t>>> load_IDX3(const std::string &filename) {
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

    // Allocate storage for images
    std::vector<std::vector<std::vector<uint8_t>>> images(
            num_images, std::vector<std::vector<uint8_t>>(num_rows, std::vector<uint8_t>(num_cols))
    );

    // Read image data
    for (int i = 0; i < num_images; ++i) {
        for (int r = 0; r < num_rows; ++r) {
            file.read(reinterpret_cast<char*>(images[i][r].data()), num_cols);
        }
    }
    file.close();
    return images;
}

uint8_t* load_IDX1_to_array(const std::string &filename, size_t &num_labels) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    // Read the header
    uint32_t magic_number = 0;
    file.read(reinterpret_cast<char *>(&magic_number), 4);

    file.read(reinterpret_cast<char *>(&num_labels), 4);

    // Convert from big-endian to host-endian
    magic_number = __builtin_bswap32(magic_number);
    num_labels = __builtin_bswap32(num_labels);

    if (magic_number != 0x00000801) {
        throw std::runtime_error("Invalid magic number in IDX1 file");
    }

    // Allocate memory for the labels
    uint8_t* labels = new uint8_t[num_labels];

    // Read the labels into the array
    file.read(reinterpret_cast<char *>(labels), num_labels);

    if (!file) {
        delete[] labels; // Free allocated memory if read fails
        throw std::runtime_error("Error reading labels from file");
    }


    return labels;
}
