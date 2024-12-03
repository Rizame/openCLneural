__kernel void initialize_weights_and_biases(
    __global double* weights,        // Buffer to store weights
    __global double* biases,         // Buffer to store biases
    int num_weights,           // Total number of weights
    int num_biases,            // Total number of biases
    unsigned int seed
) {
    int id = get_global_id(0);       // Get the global thread ID (which is unique for each thread)
    int layer_input_size = 0;
    if (id < 200704) {
        layer_input_size = 784;  // Input layer
    } else if (id < 200704 + 2560) {
        layer_input_size = 256;  // First hidden layer
    } else {
        layer_input_size = 10;   // Output layer
    }

    if (id < num_weights) {
        //double initRange = sqrt(2.0 / layer_input_size); // Standard deviation for He initialization

        weights[id] = sin((id + seed) * 12.9898);  // Random number between -1 and 1 using sine
        weights[id] = fmod(weights[id], 1.0);  // Normalize between 0 and 1

        //weights[id] = (weights[id] - 0.5) * 2.0;  // Adjust to [-1, 1]
        //weights[id] = weights[id] * initRange;  // Scale according to He initialization
        if(weights[id] < -1){
            weights[id] = -1.0;
        }
        if(weights[id] > 1){
            weights[id] = 1.0;
        }

    }

    // Initialize biases to 0
}