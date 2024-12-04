#include "../inc/NeuralNetwork.h"

std::string NeuralNetwork::read_kernel_file(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open kernel file: " << filename << std::endl;
        return "";
    }

    std::stringstream buf;
    buf << file.rdbuf();
    return buf.str();
}

void NeuralNetwork::initialize_weights_and_biases() {
// Step 1: Ensure OpenCL is initialized
    if (!context_ || !commandQueue_) {
        std::cerr << "OpenCL context or command queue not initialized!" << std::endl;
        return;
    }

    // Step 2: Create OpenCL buffers for weights and biases
    cl_int err;

    // Total number of weights and biases (weights for each layer and biases)
    int totalWeights = 0;
    int totalBiases = 0;

    // Calculate the number of weights and biases based on layer topology
    for (size_t i = 0; i < layers.size(); ++i) {
        totalWeights += layers[i].weights.size();
        totalBiases += layers[i].biases.size();
    }



    // Buffers to store weights and biases
    cl_mem weightsBuffer = clCreateBuffer(context_, CL_MEM_READ_WRITE, totalWeights * sizeof(double ), nullptr, &err);
    if (err != CL_SUCCESS || !weightsBuffer) {
        std::cerr << "Failed to create OpenCL buffer for weights." << std::endl;
        return;
    }

    cl_mem biasesBuffer = clCreateBuffer(context_, CL_MEM_READ_WRITE, totalBiases * sizeof(double ), nullptr, &err);
    if (err != CL_SUCCESS || !biasesBuffer) {
        std::cerr << "Failed to create OpenCL buffer for biases." << std::endl;
        return;
    }

    // Step 3: Prepare data for kernel execution (initialize to zero first)
    std::vector<float> zeros(totalWeights + totalBiases, 0.0f);

    // Write zero-initialized data into buffers
    err = clEnqueueWriteBuffer(commandQueue_, weightsBuffer, CL_TRUE, 0, totalWeights * sizeof(double ), zeros.data(), 0,
                               nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to write weights buffer." << std::endl;
        return;
    }

    err = clEnqueueWriteBuffer(commandQueue_, biasesBuffer, CL_TRUE, 0, totalBiases * sizeof(double ),
                               zeros.data() + totalWeights, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to write biases buffer." << std::endl;
        return;
    }

    // Step 4: Load the OpenCL kernel code and compile it
    const std::string kernelCode = read_kernel_file("src/NeuralNetworkKernel.cl"); // Read the kernel code from the file

    const char *kernelSource = kernelCode.c_str();
    cl_program program = clCreateProgramWithSource(context_, 1, &kernelSource, nullptr, &err);
    if (err != CL_SUCCESS || !program) {
        std::cerr << "Failed to create OpenCL program." << std::endl;
        return;
    }

    err = clBuildProgram(program, 1, &device_, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to build OpenCL program." << std::endl;
        size_t logSize;
        clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);

        char* log = new char[logSize];
        clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, logSize, log, nullptr);

        std::cerr << "Build log:\n" << log << std::endl;
        delete[] log;
        return;
    }

    // Step 5: Create the kernel for weight and bias initialization
    cl_kernel kernel = clCreateKernel(program, "initialize_weights_and_biases", &err);
    if (err != CL_SUCCESS || !kernel) {
        std::cerr << "Failed to create OpenCL kernel." << std::endl;
        return;
    }
    auto seed = static_cast<unsigned int>(std::time(0));

    // Step 6: Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &weightsBuffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &biasesBuffer);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &totalWeights);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &totalBiases);
    err |= clSetKernelArg(kernel, 4, sizeof(unsigned int), &seed);

    if (err != CL_SUCCESS) {
        std::cerr << "Failed to set OpenCL kernel arguments." << std::endl;
        return;
    }


    // Step 7: Launch the kernel
    size_t globalWorkSize = totalWeights + totalBiases;
    err = clEnqueueNDRangeKernel(commandQueue_, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to enqueue OpenCL kernel." << std::endl;
        return;
    }

    // Step 8: Read the data back from the buffers (weights and biases)
    std::vector<double> initializedWeights(totalWeights);
    std::vector<double> initializedBiases(totalBiases);

    err = clEnqueueReadBuffer(commandQueue_, weightsBuffer, CL_TRUE, 0, totalWeights * sizeof(float),
                              initializedWeights.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to read weights buffer." << std::endl;
        return;
    }

    err = clEnqueueReadBuffer(commandQueue_, biasesBuffer, CL_TRUE, 0, totalBiases * sizeof(float),
                              initializedBiases.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to read biases buffer." << std::endl;
        return;
    }
    clFinish(commandQueue_);

    // Step 9: Update the neural network layers with the initialized weights and biases
    size_t weightIndex = 0;
    size_t biasIndex = 0;



    for (size_t i = 1; i < layers.size(); ++i) {
        for (size_t j = 0; j < layers[i].neurons.size(); ++j) {
            for (size_t k = 0; k < layers[i-1].neurons.size(); ++k) {
                size_t flatIndex = j * layers[i-1].neurons.size() + k;
                layers[i].weights[flatIndex] = initializedWeights[weightIndex++];
            }
            layers[i].biasWeights[j] = initializedBiases[biasIndex++];
        }
    }

    //layers[i].weights[j][k] = initializedWeights[weightIndex++];

    // Step 10: Cleanup OpenCL resources
    clReleaseMemObject(weightsBuffer);
    clReleaseMemObject(biasesBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);

}


NeuralNetwork::NeuralNetwork(const std::vector<int> &topology) : platform_(nullptr), device_(nullptr),
                                                                 context_(nullptr),
                                                                 commandQueue_(nullptr) { // 784, 256, 10
    // Initialize the layers and neurons based on the topology
    openCL_init();
    for (size_t i = 0; i < topology.size(); ++i) {
        layers.push_back(Layer(topology[i], (i == 0 ? 0 : topology[i - 1]), i));
    }


    // Initialize weights and biases after constructing the layers
    initialize_weights_and_biases();
}

void NeuralNetwork::feedForward(const std::vector<double> &input) {
    cl_int err;


    for(int i = 1; i < layers.size();i++){
        std::vector<Neuron>outputNeurons{layers[i].neurons.size()};

        cl_mem output = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        layers[i].neurons.size() * sizeof(Neuron), layers[i].neurons, &err);

        cl_mem neurons = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       layers[i].neurons.size() * sizeof(Neuron), layers[i].neurons, &err);
        if (err != CL_SUCCESS) {
            std::cerr << "Error creating buffer." << std::endl;
        }
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &neurons);



        if (err != CL_SUCCESS) {
            std::cerr << "Error setting kernel argument." << std::endl;
        }
        size_t globalWorkSize = layers[i].neurons.size();

        err = clEnqueueNDRangeKernel(commandQueue, kernel, 1, nullptr,
                                     &globalWorkSize, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to enqueue OpenCL kernel." << std::endl;
            return;
        }

        err = clEnqueueReadBuffer(commandQueue_, output, CL_TRUE, 0,  layers[i].neurons.size() * sizeof(Neuron),
                                  outputNeurons.data(), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to read weights buffer." << std::endl;
            return;
        }
        for(int k = 0; k < layers[i].neurons.size();k++){
            layers[i].neurons[k].value = outputNeurons[k].value;
        }
        clReleaseMemObject(output);
        clReleaseMemObject(neurons);

    }

}

void NeuralNetwork::backPropagate(const std::vector<double> &target) {

}

void
NeuralNetwork::train(const std::vector<std::vector<double>> &inputs, const std::vector<std::vector<double>> &targets,
                     int epochs, double learningRate) {

}

bool NeuralNetwork::openCL_init() {
    cl_int err;

    // Step 1: Get the number of platforms available
    cl_uint platformCount = 0;
    err = clGetPlatformIDs(0, nullptr, &platformCount);
    if (err != CL_SUCCESS || platformCount == 0) {
        std::cerr << "Failed to get OpenCL platform count or no platforms available." << std::endl;
        return false;
    }

    // Step 2: Get platform IDs
    std::vector<cl_platform_id> platforms(platformCount);
    err = clGetPlatformIDs(platformCount, platforms.data(), nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get OpenCL platform IDs." << std::endl;
        return false;
    }

    // Step 3: Select a platform (for simplicity, choose the first one)
    platform_ = platforms[0];

    // Step 4: Get the number of devices for the selected platform
    cl_uint deviceCount = 0;
    err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceCount);
    if (err != CL_SUCCESS || deviceCount == 0) {
        std::cerr << "Failed to get OpenCL device count or no devices available." << std::endl;
        return false;
    }

    // Step 5: Get device IDs (we choose the first one for simplicity)
    std::vector<cl_device_id> devices(deviceCount);
    err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, deviceCount, devices.data(), nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get OpenCL device IDs." << std::endl;
        return false;
    }

    device_ = devices[0];

    // Step 6: Create an OpenCL context
    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    if (err != CL_SUCCESS || !context_) {
        std::cerr << "Failed to create OpenCL context." << std::endl;
        return false;
    }

    commandQueue_ = clCreateCommandQueue(context_, device_, 0, &err);
    if (err != CL_SUCCESS || !commandQueue_) {
        std::cerr << "Failed to create OpenCL command queue." << std::endl;
        return false;
    }

    return true;
}


