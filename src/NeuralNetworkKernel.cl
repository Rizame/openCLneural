__kernel void initialize_weights_and_biases(
    __global double* weights,        // Buffer to store weights
    __global double* biases,         // Buffer to store biases
    const int num_weights,           // Total number of weights
    const int num_biases,            // Total number of biases
    const int layer_input_size       // The size of the layer before this one (for He initialization)
) {
    int id = get_global_id(0);       // Get the global thread ID (which is unique for each thread)

    // He initialization for weights
    if (id < num_weights) {
        double initRange = sqrt((double)(2.0 / layer_input_size)); // Standard deviation for He initialization
        weights[id] = (fract(sin(id * 12.9898) * 43758.5453)) * initRange;  // Random weight initialization
    }

    // Initialize biases to 0
    if (id < num_biases) {
        biases[id] = 0.0; // Biases are initialized to 0
    }
}