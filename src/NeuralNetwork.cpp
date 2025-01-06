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

    size_t globalWorkSize = totalWeights + totalBiases;

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
    cl_mem seedsBuffer = createReadBufferFromVector(seeds, CL_MEM_READ_ONLY);


    // Step 5: Create the kernel for weight and bias initialization
    cl_kernel kernel = clCreateKernel(program, "init", &err);
    if (err != CL_SUCCESS || !kernel) {
        std::cerr << "Failed to create OpenCL kernel." << std::endl;
        return;
    }

    // Step 6: Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &weightsBuffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &biasesBuffer);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &seedsBuffer);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &totalWeights);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &totalBiases);

    if (err != CL_SUCCESS) {
        std::cerr << "Failed to set OpenCL kernel arguments." << std::endl;
        return;
    }

    // Step 7: Launch the kernel
    err = clEnqueueNDRangeKernel(commandQueue_, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to enqueue OpenCL kernel." << std::endl;
        return;
    }


    clFinish(commandQueue_);


    // Step 10: Cleanup OpenCL resources

    clReleaseKernel(kernel);
}

void NeuralNetwork::initialize_topology_buffer(const std::vector<int> &topology) {
    cl_int err;
    totalLayers = static_cast<int>(topology.size());

    // Create buffer for topology
    topologyBuffer = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    topology.size() * sizeof(int),
                                    const_cast<int *>(topology.data()), &err);

    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create topology buffer!" << std::endl;
        return;
    }
}

NeuralNetwork::NeuralNetwork(const std::vector<int> &topology) : totalWeights(0), totalBiases(0), totalNeurons(0),
                                                                 totalDeltas(0), platform_(nullptr), device_(nullptr),
                                                                 context_(nullptr), commandQueue_(nullptr) {

    for (size_t i = 0; i < topology.size(); ++i) {
        layers.emplace_back(topology[i], (i == 0 ? 0 : topology[i - 1]), i);
    }


    for (size_t i = 0; i < layers.size(); ++i) {
        totalWeights += layers[i].weights.size();
        totalBiases += layers[i].biases.size();
        totalNeurons += layers[i].neurons.size();
        totalDeltas += layers[i].deltas.size();
    }

    openCL_init();

    // Initialize weights and biases after constructing the layers
    initialize_weights_and_biases();

    // Initialize topology buffer for OpenCL
    initialize_topology_buffer(topology);
}

void NeuralNetwork::feedForward(std::vector<double> &input) {
    if (!context_ || !commandQueue_) {
        std::cerr << "OpenCL context or command queue not initialized!" << std::endl;
        return;
    }
    std::vector<Neuron> inputNeurons(layers[0].neurons.size());
    for (size_t j = 0; j < input.size(); j++) {
        inputNeurons[j].value = input[j];
    }

    cl_int err;
    err = clEnqueueWriteBuffer(commandQueue_, neuronsBuffer, CL_TRUE, 0, inputNeurons.size() * sizeof(Neuron),
                               inputNeurons.data(), 0,
                               nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Error setting input buffer." << std::endl;
    }

    err = clSetKernelArg(kernelFF, 0, sizeof(cl_mem), &neuronsBuffer);
    err |= clSetKernelArg(kernelFF, 1, sizeof(cl_mem), &biasesBuffer);
    err |= clSetKernelArg(kernelFF, 2, sizeof(cl_mem), &weightsBuffer);
    err |= clSetKernelArg(kernelFF, 3, sizeof(cl_mem), &topologyBuffer);

    if (err != CL_SUCCESS) {
        std::cerr << "Error setting kernel FF basic arguments." << std::endl;
    }
    int offset_n = 0;

    for (int i = 1; i < layers.size(); i++) {
        err = clSetKernelArg(kernelFF, 4, sizeof(int), &i);


        if (err != CL_SUCCESS) {
            std::cerr << "Error setting kernel FF layer argument." << std::endl;
        }
        size_t globalWorkSize = layers[i].neurons.size();

        err = clEnqueueNDRangeKernel(commandQueue_, kernelFF, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr,
                                     nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to enqueue OpenCL kernel." << std::endl;
            return;
        }

        err = clEnqueueReadBuffer(commandQueue_, neuronsBuffer, CL_TRUE,
                                  ( offset_n + layers[0].neurons.size()) * sizeof(double),
                                  layers[i].neurons.size() * sizeof(double),
                                  layers[i].neurons.data(), 0, nullptr, nullptr);

        if (err != CL_SUCCESS) {
            std::cerr << "Failed to read neuron buffer. ERR code:" << std::endl;
            return;
        }

//        size_t sum = 0;
//        for (int j = i - 1; j >= 0; j--) {
//            sum += layers[j].weights.size();
//        }
//        err = clEnqueueReadBuffer(commandQueue_, weightsBuffer, CL_TRUE,
//                                  sum * sizeof(double), layers[i].weights.size() * sizeof(double),
//                                  layers[i].weights.data(), 0,
//                                  nullptr, nullptr);

        offset_n += layers[i].neurons.size();

        if (err != CL_SUCCESS) {
            std::cerr << "Failed to read weights buffer. ERR code:" << std::endl;
            return;
        }

        clFinish(commandQueue_);

    }
    double guessVal = 0.0;
    for (int i = 0; i < layers.back().neurons.size(); i++) {
        double value = layers.back().neurons[i].value;
        if (value > guessVal) {
            guess = i;
            guessVal = static_cast<double>(value);
        }
    }

}

void NeuralNetwork::Debug(std::ofstream &filename) {
    for (int i = 0; i < layers.size(); i++) {
        filename << "\n";
        for (int j = 0; j < layers[i].weights.size(); j++) {
            filename << layers[i].weights[j] << ',';
        }
    }
}

void NeuralNetwork::backPropagate(int target) {
    if (!context_ || !commandQueue_) {
        std::cerr << "OpenCL context or command queue not initialized!" << std::endl;
        return;
    }
    double learningRate = 0.001;

    cl_int err;


    // Iterate over layers in reverse order
    for (size_t layer = layers.size() - 1; layer > 0; layer--) {

        // Set kernel arguments
        int isOutputLayer = (layer == layers.size() - 1);

        err = clSetKernelArg(kernelBP, 0, sizeof(cl_mem), &neuronsBuffer);
        err |= clSetKernelArg(kernelBP, 1, sizeof(cl_mem), &weightsBuffer);
        err |= clSetKernelArg(kernelBP, 2, sizeof(cl_mem), &deltasBuffer);
        err |= clSetKernelArg(kernelBP, 3, sizeof(cl_mem), &biasesBuffer);
        err |= clSetKernelArg(kernelBP, 4, sizeof(cl_mem), &topologyBuffer);
        err |= clSetKernelArg(kernelBP, 5, sizeof(int), &isOutputLayer);
        err |= clSetKernelArg(kernelBP, 6, sizeof(int), &target);
        err |= clSetKernelArg(kernelBP, 7, sizeof(int), &layer);
        err |= clSetKernelArg(kernelBP, 8, sizeof(double), &learningRate);

        if (err != CL_SUCCESS) {
            std::cerr << "Error setting argument." << std::endl;
        }


        size_t globalWorkSize = layers[layer].neurons.size();

        // Run kernel
        err = clEnqueueNDRangeKernel(commandQueue_, kernelBP, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr,
                                     nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to enqueue OpenCL kernel." << std::endl;
            return;
        }
        int offset_n = 0;
        layer == 2 ? offset_n = 784*256 : 0;
        err = clEnqueueReadBuffer(commandQueue_, weightsBuffer, CL_TRUE,
                                  ( offset_n) * sizeof(double ),
                                  layers[layer].weights.size() * sizeof(double ),
                                  layers[layer].weights.data(), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to read weights." << std::endl;
            return;
        }

        clFinish(commandQueue_);

    }


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
    const std::string kernelCode = read_kernel_file("src/kernelFn.cl");
    kernelSource = kernelCode.c_str();

    program = clCreateProgramWithSource(context_, 1, &kernelSource, nullptr, &err);
    if (err != CL_SUCCESS || !program) {
        std::cerr << "Failed to create OpenCL program." << std::endl;
        return false;
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
        return false;
    }

    weightsBuffer = createWriteBuffer<double>(totalWeights);
    biasesBuffer = createWriteBuffer<double>(totalBiases);
    neuronsBuffer = createWriteBuffer<Neuron>(totalNeurons);
    deltasBuffer = createWriteBuffer<double>(totalDeltas);


    kernelFF = clCreateKernel(program, "feed_forward", &err);
    if (err != CL_SUCCESS || !kernelFF) {
        std::cerr << "Failed to create OpenCL kernel." << std::endl;
        return false;
    }
    kernelBP = clCreateKernel(program, "back_propagation", &err);
    if (err != CL_SUCCESS || !kernelBP) {
        std::cerr << "Failed to create OpenCL kernel." << std::endl;
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
    avg_error = 0;
    for (size_t i = 0; i < layers.back().neurons.size(); i++) {
        avg_error += error(layers.back().neurons[i].value, target == i);
    }
    avg_error /= (double) layers.back().neurons.size();
    std::cout << "\nThe activated neuron value: " << layers.back().neurons[target].value << std::endl;
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

NeuralNetwork::~NeuralNetwork() {
    clReleaseMemObject(neuronsBuffer);
    clReleaseMemObject(topologyBuffer);
    clReleaseMemObject(weightsBuffer);
    clReleaseMemObject(deltasBuffer);
    clReleaseMemObject(biasesBuffer);
    clReleaseKernel(kernelBP);
    clReleaseKernel(kernelFF);
}
