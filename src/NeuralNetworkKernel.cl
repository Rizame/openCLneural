double rand(ulong* c) {
     ulong MODULUS = 281474976710656ULL; // 2^48
     ulong MULTIPLIER = 19073486328125ULL;

    // Update the seed
    *c = (*c * MULTIPLIER + 1) % MODULUS;
    return (double)(*c) / (double)MODULUS;
}

__kernel void initialize_weights_and_biases(
    __global double* weights,        // Buffer to store weights
    __global double* biases,         // Buffer to store biases
    int num_weights,           // Total number of weights
    int num_biases,            // Total number of biases
    __global double* seeds
) {
    int id = get_global_id(0);       // Get the global thread ID (which is unique for each thread)

    if (id < num_weights) {

         weights[id] = sin((id + seeds[id]) * 5.1928667898);  // Random number between -1 and 1 using sine
                weights[id] = fmod(weights[id], 1.0); // Normalize between 0 and 1


        if(weights[id] < -1){
            weights[id] = -1.0;
        }
        if(weights[id] > 1){
            weights[id] = 1.0;
        }

    }
    if (id < num_biases){
        biases[id] = sin((id + seeds[id]) * 112.74932);  // Random number between -1 and 1 using sine
        biases[id] = fmod(biases[id], (double)1.0);  // Normalize between 0 and 1

        if(biases[id] < -1){
            biases[id] = -1.0;
        }
        if(biases[id] > 1){
            biases[id] = 1.0;
        }
    }

    // Initialize biases to 0
}