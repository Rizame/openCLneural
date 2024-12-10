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
    seeds.resize(totalWeights+totalBiases,0.0);

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    std::default_random_engine generator(seed);
    std::uniform_int_distribution<long> distribution(1, RAND_MAX);
    for (int i = 0; i < totalWeights+totalBiases; ++i) {
        double random_number = distribution(generator);
        random_number /= 100;
        seeds[i] = random_number;
    }




    // Buffers to store weights and biases
    cl_mem weightsBuffer = clCreateBuffer(context_, CL_MEM_READ_WRITE, totalWeights * sizeof(double), nullptr, &err);
    if (err != CL_SUCCESS || !weightsBuffer) {
        std::cerr << "Failed to create OpenCL buffer for weights." << std::endl;
        return;
    }

    cl_mem seedsBuff = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, (totalWeights+totalBiases) * sizeof(double), seeds.data(), &err);
    if (err != CL_SUCCESS || !seedsBuff) {
        std::cerr << "Failed to create OpenCL buffer for seeds." << std::endl;
        return;
    }

    cl_mem biasesBuffer = clCreateBuffer(context_, CL_MEM_READ_WRITE, totalBiases * sizeof(double), nullptr, &err);
    if (err != CL_SUCCESS || !biasesBuffer) {
        std::cerr << "Failed to create OpenCL buffer for biases." << std::endl;
        return;
    }

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
                if(initializedWeights[weightIndex] == 0 && weightIndex < 130000) printf("weight:%zu\n",weightIndex);
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

    for(int i = 0; i < layers.size();i++){

        int neuronsSize = static_cast<int>(layers[i].neurons.size());

        cl_mem outputBuff = clCreateBuffer(context_, CL_MEM_READ_WRITE,
                                           neuronsSize * sizeof(Neuron), nullptr, &err);

        if (err != CL_SUCCESS) {
            std::cerr << "Error creating output buffer." << std::endl;
            return;
        }

        cl_mem inputBuff = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            i == 0 ? (input.size() * sizeof(double)) : 1, input.data(), &err);
        if (err != CL_SUCCESS) {
            std::cerr << "Error creating input buffer." << std::endl;
            return;
        }

        cl_mem neuronsBuff = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            i == 0 ? 1 : (layers[i-1].neurons.size() * sizeof(Neuron)), i == 0 ? layers[i].neurons.data() : layers[i-1].neurons.data(), &err);
        if (err != CL_SUCCESS) {
            std::cerr << "Error creating buffer." << std::endl;
            return;
        }

        cl_mem weightsBuff = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        i != 0 ? layers[i].weights.size() * sizeof(double): 1, layers[i].weights.data(), &err);
        if (err != CL_SUCCESS) {
            std::cerr << "Error creating weights buffer." << std::endl;
        }

        cl_mem biasweightsBuff = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                i != 0 ? layers[i].biasWeights.size() : 1 * sizeof(double), layers[i].biasWeights.data(), &err);

        if (err != CL_SUCCESS) {
            std::cerr << "Error creating bias weights buffer." << std::endl;
        }
        cl_mem BiasBuff = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         i != 0 ? layers[i].biases.size() : 1 * sizeof(double), layers[i].biases.data(), &err);

        if (err != CL_SUCCESS) {
            std::cerr << "Error creating bias buffer." << std::endl;
            return;
        }

        int prevLayerNeuron = i == 0 ? 0 : static_cast<int>(layers[i-1].neurons.size());

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

        err = clEnqueueNDRangeKernel(commandQueue_, kernel, 1, nullptr,
                                     &globalWorkSize, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to enqueue OpenCL kernel." << std::endl;
            return;
        }

        err = clEnqueueReadBuffer(commandQueue_, outputBuff, CL_TRUE, 0,  neuronsSize * sizeof(Neuron),
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
    normalizeBeforeSoft(-2,2);

    //layers[layers.size()-1].neurons is the last layer of neurons after the weighted sum calculation.
    auto maxIt = std::max_element(layers[layers.size()-1].neurons.begin(), layers[layers.size()-1].neurons.end(),
                                  [](const Neuron& a, const Neuron& b) {
                                      return a.value < b.value;
                                  });
    double maxValue = maxIt != layers[layers.size()-1].neurons.end() ? maxIt->value : 0.0;

    double exp_sum = std::accumulate(layers[layers.size()-1].neurons.begin(), layers[layers.size()-1].neurons.end(), 0.0,
                                 [maxValue](double total, const Neuron& neuron) {
                                     return total + exp(neuron.value - maxValue);
                                 });


    double guessVal = 0.0;
    for(int i = 0; i < layers[layers.size()-1].neurons.size();i++){
        double value = exp(layers[layers.size()-1].neurons[i].value - maxValue)/exp_sum;
        layers[layers.size()-1].neurons[i].value = static_cast<double>(value);
        if (value > guessVal){
            guess = i;
            guessVal = static_cast<double>(value);
        }
    }

    std::cout<<"Guessed: "<<guess<<std::endl;

    clReleaseKernel(kernel);
    clReleaseProgram(program);
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
void NeuralNetwork::errorCalculation(int target) {
    avg_error = -log(std::max(layers[layers.size()-1].neurons[target].value, 1e-7));

    std::cout<<"\nThe activated neuron value: "<<layers[layers.size()-1].neurons[target].value<<std::endl;
    std::cout<<"\nCalculated error: "<<avg_error<<std::endl;
}
void NeuralNetwork::normalizeBeforeSoft(double lowerBound, double upperBound) {

    auto maxIt = std::max_element(layers[layers.size()-1].neurons.begin(), layers[layers.size()-1].neurons.end(),
                                  [](const Neuron& a, const Neuron& b) {
                                      return a.value < b.value;
                                  });
    double maxVal = maxIt->value;
    auto minIt = std::min_element(layers[layers.size()-1].neurons.begin(), layers[layers.size()-1].neurons.end(),
                                  [](const Neuron& a, const Neuron& b) {
                                      return a.value < b.value;
                                  });
    double minVal = minIt->value;

    std::vector<double> normalizedValues;

    for (int i = 0; i < layers[layers.size()-1].neurons.size();i++) {
        double norm = lowerBound + (upperBound - lowerBound) * (layers[layers.size()-1].neurons[i].value - minVal) / (maxVal - minVal);

        // Clamp values slightly inside the range to avoid strict boundaries
        if (norm <= lowerBound) norm = lowerBound + 0.01;
        if (norm >= upperBound) norm = upperBound - 0.01;

        layers[layers.size()-1].neurons[i].value = norm;
    }
}



