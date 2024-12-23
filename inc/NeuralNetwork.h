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
    int guess = -1;

    NeuralNetwork(const std::vector<int> &topology);

    void initialize_weights_and_biases();

    bool openCL_init();

    void feedForward(std::vector<double> &input);

    void backPropagate(int target);

    void errorCalculation(int target);

    void normalizeBeforeSoft(double lowerBound, double upperBound);

    std::string read_kernel_file(const std::string &filename);

private:
    int totalWeights;
    int totalBiases;
    int totalNeurons;
    int totalDeltas;

    std::vector<Layer> layers;
    double avg_error = 0.0;

    const char *kernelSource;

    cl_program program;

    cl_mem weightsBuffer;
    cl_mem biasesBuffer;
    cl_mem neuronsBuffer;
    cl_mem deltasBuffer;

    cl_kernel kernelFF;
    cl_kernel kernelBP;

    cl_platform_id platform_;      // OpenCL platform
    cl_device_id device_;          // OpenCL device
    cl_context context_;           // OpenCL context
    cl_command_queue commandQueue_; // Command queue for the device

    template<typename T>
    cl_mem createReadBufferFromVector(std::vector<T> &input, cl_mem_flags flags) {
        cl_int err = CL_SUCCESS;
        cl_mem buff = clCreateBuffer(context_, flags | CL_MEM_USE_HOST_PTR, input.size() * sizeof(T), static_cast<void *>(input.data()),
                                     &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error{"Error creating input buffer"};
        }
        return buff;
    }

    template<typename T>
    cl_mem createWriteBuffer(size_t size) {
        cl_int err = CL_SUCCESS;
        cl_mem buff = clCreateBuffer(context_, CL_MEM_READ_WRITE, size * sizeof(T), nullptr, &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error{"Error creating output buffer"};
        }
        return buff;
    }
};


#endif //NEURALDIGITRECON_NEURALNETWORK_H
