#include <cmath>
#include <random>
#include <chrono>
#include <numeric>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include "Layer.h"
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#ifndef NEURALDIGITRECON_NEURALNETWORK_H
#define NEURALDIGITRECON_NEURALNETWORK_H


class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& topology);
    void initialize_weights_and_biases();
    bool openCL_init();
    void feedForward(std::vector<double>& input);
    void backPropagate(const std::vector<double>& target);
    void train(const std::vector<std::vector<double>>& inputs,
               const std::vector<std::vector<double>>& targets,
               int epochs, double learningRate);
    void errorCalculation(int target);
    void normalizeBeforeSoft(double lowerBound, double upperBound);
    std::string read_kernel_file(const std::string& filename);
private:
    std::vector<Layer> layers;
    double avg_error = 0.0;
    int guess = -1;

    cl_platform_id platform_;      // OpenCL platform
    cl_device_id device_;          // OpenCL device
    cl_context context_;           // OpenCL context
    cl_command_queue commandQueue_; // Command queue for the device
};


#endif //NEURALDIGITRECON_NEURALNETWORK_H
