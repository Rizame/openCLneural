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
    for (size_t i = 0; i < layers.size(); ++i) {
        totalWeights += layers[i].weights.size();
        totalBiases += layers[i].biases.size();
    }

    std::vector<double> seeds{};
    seeds.resize(totalWeights + totalBiases, 0.0);

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    std::default_random_engine generator(seed);
    std::uniform_int_distribution<long> distribution(1, RAND_MAX);
    for (int i = 0; i < totalWeights + totalBiases; ++i) {
        double random_number = distribution(generator);
        random_number /= 100;
        seeds[i] = random_number;
    }


    // Buffers to store weights and biases
    cl_mem weightsBuffer = createWriteBuffer<double>(totalWeights);
    cl_mem seedsBuff = createReadBufferFromVector(seeds, CL_MEM_READ_ONLY);
    cl_mem biasesBuffer = createWriteBuffer<double>(totalBiases);

    // Step 3: Prepare data for kernel execution (initialize to zero first)
    std::vector<double> zeros(totalWeights + totalBiases, 0.0);

    // Write zero-initialized data into buffers
    err = clEnqueueWriteBuffer(commandQueue_, weightsBuffer, CL_TRUE, 0, totalWeights * sizeof(double), zeros.data(), 0,
                               nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to write weights buffer." << std::endl;
        return;
    }

    err = clEnqueueWriteBuffer(commandQueue_, biasesBuffer, CL_TRUE, 0, totalBiases * sizeof(double),
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

        char *log = new char[logSize];
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
    //auto seed = static_cast<uint64_t >(std::time(0));

    // Step 6: Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &weightsBuffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &biasesBuffer);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &totalWeights);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &totalBiases);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &seedsBuff);

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

    err = clEnqueueReadBuffer(commandQueue_, weightsBuffer, CL_TRUE, 0, totalWeights * sizeof(double),
                              initializedWeights.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to read weights buffer." << std::endl;
        return;
    }

    err = clEnqueueReadBuffer(commandQueue_, biasesBuffer, CL_TRUE, 0, totalBiases * sizeof(double),
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
            for (size_t k = 0; k < layers[i - 1].neurons.size(); ++k) {
                size_t flatIndex = j * layers[i - 1].neurons.size() + k;
                layers[i].weights[flatIndex] = initializedWeights[weightIndex++];
                if (initializedWeights[weightIndex] == 0 && weightIndex < 130000) printf("weight:%zu\n", weightIndex);
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
                                                                 context_(nullptr), commandQueue_(nullptr) {
    // 784, 256, 10
    // Initialize the layers and neurons based on the topology
    openCL_init();

    auto sigmoid = [](double x) {
        return 1 / (1 + exp(-x));
    };

    auto sigmoid_deriv = [=](double x) {
        double val = sigmoid(x);
        return val * (1 - val);
    };

    for (size_t i = 0; i < topology.size(); ++i) {
        layers.emplace_back(topology[i], (i == 0 ? 0 : topology[i - 1]), i, sigmoid, sigmoid_deriv);
    }


    // Initialize weights and biases after constructing the layers
    initialize_weights_and_biases();
}

void NeuralNetwork::feedForward(std::vector<double> &input) {
    if (!context_ || !commandQueue_) {
        std::cerr << "OpenCL context or command queue not initialized!" << std::endl;
        return;
    }

    cl_int err;

    const std::string kernelCode = read_kernel_file("src/feedForward.cpp"); // Read the kernel code from the file
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

        char *log = new char[logSize];
        clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, logSize, log, nullptr);

        std::cerr << "Build log:\n" << log << std::endl;
        delete[] log;
        return;
    }

    // Step 5: Create the kernel for weight and bias initialization
    cl_kernel kernel = clCreateKernel(program, "feed_forward_cl", &err);
    if (err != CL_SUCCESS || !kernel) {
        std::cerr << "Failed to create OpenCL kernel." << std::endl;
        return;
    }

    for (int i = 0; i < layers.size(); i++) {
        int neuronsSize = static_cast<int>(layers[i].neurons.size());

        cl_mem outputBuff = createWriteBuffer<Neuron>(layers[i].neurons.size());

        if (err != CL_SUCCESS) {
            std::cerr << "Error creating output buffer." << std::endl;
            return;
        }

        cl_mem inputBuff = createReadBufferFromVector(input, CL_MEM_READ_ONLY);

        cl_mem neuronsBuff = createReadBufferFromVector(layers[i == 0 ? i : i - 1].neurons, CL_MEM_READ_ONLY);


        cl_mem weightsBuff = createReadBufferFromVector(layers[i].weights, CL_MEM_READ_ONLY);

        cl_mem biasweightsBuff = createReadBufferFromVector(layers[i].biasWeights, CL_MEM_READ_ONLY);

        cl_mem BiasBuff = createReadBufferFromVector(layers[i].biases, CL_MEM_READ_ONLY);

        int prevLayerNeuron = i == 0 ? 0 : static_cast<int>(layers[i - 1].neurons.size());

        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &neuronsBuff);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &BiasBuff);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &biasweightsBuff);
        err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &inputBuff);
        err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &weightsBuff);
        err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &outputBuff);
        err |= clSetKernelArg(kernel, 6, sizeof(int), &prevLayerNeuron);
        err |= clSetKernelArg(kernel, 7, sizeof(int), &i);

        if (err != CL_SUCCESS) {
            std::cerr << "Error setting kernel argument." << std::endl;
        }
        size_t globalWorkSize = neuronsSize;

        err = clEnqueueNDRangeKernel(commandQueue_, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to enqueue OpenCL kernel." << std::endl;
            return;
        }

        err = clEnqueueReadBuffer(commandQueue_, outputBuff, CL_TRUE, 0, neuronsSize * sizeof(Neuron),
                                  layers[i].neurons.data(), 0, nullptr, nullptr);

        clFinish(commandQueue_);

        if (err != CL_SUCCESS) {
            std::cerr << "Failed to read weights buffer." << std::endl;
            return;
        }


        clReleaseMemObject(outputBuff);
        clReleaseMemObject(neuronsBuff);
        clReleaseMemObject(inputBuff);
        clReleaseMemObject(weightsBuff);
        clReleaseMemObject(biasweightsBuff);
        clReleaseMemObject(BiasBuff);
    }


    double guessVal = 0.0;
    for (int i = 0; i < layers[layers.size() - 1].neurons.size(); i++) {
        //double value = exp(layers[layers.size() - 1].neurons[i].value - maxValue) / exp_sum;  SOFTMAX
        //double value = layers[layers.size()-1].activate(layers[layers.size() - 1].neurons[i].value); // SIGMOID
        //layers[layers.size() - 1].neurons[i].value = static_cast<double>(value);
        double value = layers[layers.size() - 1].neurons[i].value;
        if (value > guessVal) {
            guess = i;
            guessVal = static_cast<double>(value);
        }
    }

//    std::cout << "\nGuessed: " << guess << std::endl;

    clReleaseKernel(kernel);
    clReleaseProgram(program);
}


void NeuralNetwork::backPropagate(int target) {
    if (!context_ || !commandQueue_) {
        std::cerr << "OpenCL context or command queue not initialized!" << std::endl;
        return;
    }
    double learningRate = 0.01;

    cl_int err;

    // Load and build the backpropagation kernel
    const std::string kernelCode = read_kernel_file("src/backPropagation.cpp");
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

        char *log = new char[logSize];
        clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, logSize, log, nullptr);

        std::cerr << "Build log:\n" << log << std::endl;
        delete[] log;
        return;
    }

    cl_kernel kernel = clCreateKernel(program, "back_propagation_cl", &err);
    if (err != CL_SUCCESS || !kernel) {
        std::cerr << "Failed to create back_prop kernel." << std::endl;
        return;
    }


    // Iterate over layers in reverse order
    for (size_t layer = layers.size() - 1; layer > 0; layer--) {
        int numCurrentNeurons = layers[layer].neurons.size();
        int numPrevNeurons = layers[layer - 1].neurons.size();
        int numNextNeurons = layer != layers.size() - 1 ? layers[layer + 1].neurons.size() : 0;

        // Create buffers
        cl_mem prevLayerValues = createReadBufferFromVector(layers[layer - 1].neurons, CL_MEM_READ_ONLY);
        cl_mem currentLayerValues = createReadBufferFromVector(layers[layer].neurons, CL_MEM_READ_ONLY);
        cl_mem weights = createReadBufferFromVector(layers[layer].weights, CL_MEM_READ_WRITE);
        cl_mem weightsNext = createReadBufferFromVector(layers[layer != layers.size() - 1 ? layer + 1 : layer].weights, CL_MEM_READ_ONLY);
        cl_mem biasWeights = createReadBufferFromVector(layers[layer].biasWeights, CL_MEM_READ_WRITE);
        cl_mem deltas = createWriteBuffer<double>(layers[layer].deltas.size());
        cl_mem nextLayerDeltas = createReadBufferFromVector(
                layers[layer != layers.size() - 1 ? layer + 1 : layer].deltas, CL_MEM_READ_ONLY);

        // Set kernel arguments
        int isOutputLayer = (layer == layers.size() - 1);

        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &prevLayerValues);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &currentLayerValues);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &weights);
        err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &weightsNext);
        err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &nextLayerDeltas);
        err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &deltas);
        err |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &biasWeights);
        err |= clSetKernelArg(kernel, 7, sizeof(int), &numPrevNeurons);
        err |= clSetKernelArg(kernel, 8, sizeof(int), &numCurrentNeurons);
        err |= clSetKernelArg(kernel, 9, sizeof(int), &numNextNeurons);
        err |= clSetKernelArg(kernel, 10, sizeof(double), &learningRate);
        err |= clSetKernelArg(kernel, 11, sizeof(int), &isOutputLayer);
        err |= clSetKernelArg(kernel, 12, sizeof(int), &target);

        if (err != CL_SUCCESS) {
            std::cerr << "Error setting argument." << std::endl;
        }


        size_t globalWorkSize = numCurrentNeurons;

        // Run kernel
        err = clEnqueueNDRangeKernel(commandQueue_, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to enqueue OpenCL kernel." << std::endl;
            return;
        }


        // Read updated weights back to host
        err = clEnqueueReadBuffer(commandQueue_, weights, CL_TRUE, 0,
                                  layers[layer].weights.size() * sizeof(double), layers[layer].weights.data(), 0,
                                  nullptr, nullptr);

        if (err != CL_SUCCESS) {
            std::cerr << "Error reading from updated weights buffer." << std::endl;
        }


        err = clEnqueueReadBuffer(commandQueue_, deltas, CL_TRUE, 0, layers[layer].deltas.size() * sizeof(double),
                                  layers[layer].deltas.data(), 0, nullptr, nullptr);

        if (err != CL_SUCCESS) {
            std::cerr << "Error reading from deltas buffer." << std::endl;
        }

        err = clEnqueueReadBuffer(commandQueue_, biasWeights, CL_TRUE, 0, layers[layer].biasWeights.size() * sizeof(double),
                                  layers[layer].biasWeights.data(), 0, nullptr, nullptr);

        clFinish(commandQueue_);

        clReleaseMemObject(prevLayerValues);
        clReleaseMemObject(currentLayerValues);
        clReleaseMemObject(weights);
        clReleaseMemObject(weightsNext);
        clReleaseMemObject(biasWeights);
        clReleaseMemObject(deltas);
        clReleaseMemObject(nextLayerDeltas);
    }

    clReleaseKernel(kernel);
    clReleaseProgram(program);


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
double error(double value, double target) {
    return (value - target) * (value - target);
}

double error_deriv(double value, double target) {
    return value - target;
}

void NeuralNetwork::errorCalculation(int target) {
    //avg_error = -log(std::max(layers[layers.size() - 1].neurons[target].value, 1e-7));
    avg_error = 0;
    for (size_t i = 0; i < layers.back().neurons.size(); i++) {
        avg_error += error(layers.back().neurons[i].value, target == i);
    }
    avg_error /= (double) layers.back().neurons.size();
    std::cout << "\nThe activated neuron value: " << layers[layers.size() - 1].neurons[target].value << std::endl;
    std::cout << "\nCalculated error: " << avg_error << std::endl;
}

void NeuralNetwork::normalizeBeforeSoft(double lowerBound, double upperBound) {
    auto maxIt = std::max_element(layers[layers.size() - 1].neurons.begin(), layers[layers.size() - 1].neurons.end(),
                                  [](const Neuron &a, const Neuron &b) {
                                      return a.value < b.value;
                                  });
    double maxVal = maxIt->value;
    auto minIt = std::min_element(layers[layers.size() - 1].neurons.begin(), layers[layers.size() - 1].neurons.end(),
                                  [](const Neuron &a, const Neuron &b) {
                                      return a.value < b.value;
                                  });
    double minVal = minIt->value;

    std::vector<double> normalizedValues;

    for (int i = 0; i < layers[layers.size() - 1].neurons.size(); i++) {
        double norm = lowerBound + (upperBound - lowerBound) * (layers[layers.size() - 1].neurons[i].value - minVal) /
                                   (maxVal - minVal);

        // Clamp values slightly inside the range to avoid strict boundaries
        if (norm <= lowerBound) norm = lowerBound + 0.01;
        if (norm >= upperBound) norm = upperBound - 0.01;

        layers[layers.size() - 1].neurons[i].value = norm;
    }
}
