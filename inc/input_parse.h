//
// Created by Anton Bucataru on 25.11.2024.
//

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <string>

// Function to read a 32-bit integer in big-endian format
int32_t read_int(std::ifstream &file);

// Function to load IDX3 file and parse images
std::vector<std::vector<double>> load_IDX3(const std::string &filename);

std::vector<int> load_IDX1_to_array(const std::string &filename, size_t num_labels);