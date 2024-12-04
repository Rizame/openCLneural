__kernel void initialize_weights_and_biases(
    __global double* weights,        // Buffer to store weights
    __global double* biases,         // Buffer to store biases
    int num_weights,           // Total number of weights
    int num_biases,            // Total number of biases
    unsigned int seed
) {
    int id = get_global_id(0);       // Get the global thread ID (which is unique for each thread)
    if(id == 0) printf("total weights: %d\n", num_weights);

    if (id < num_weights) {


        weights[id] = sin((id + seed) * 12.9898);  // Random number between -1 and 1 using sine
        weights[id] = fmod(weights[id], 1.0);  // Normalize between 0 and 1

        if(id > 203260) printf("weight value for layer 3: %f\n", weights[id]);

        if(weights[id] < -1){
            weights[id] = -1.0;
        }
        if(weights[id] > 1){
            weights[id] = 1.0;
        }
    }
    if (id < num_biases){
        biases[id] = sin((id + seed) * 11.74932);  // Random number between -1 and 1 using sine
        biases[id] = fmod(biases[id], 1.0);  // Normalize between 0 and 1

        if(biases[id] < -1){
            biases[id] = -1.0;
        }
        if(biases[id] > 1){
            biases[id] = 1.0;
        }
    }

    // Initialize biases to 0
}