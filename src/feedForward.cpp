#pragma pack(push, 1)
struct Neuron {
    double value;
    double gradient;
};
#pragma pack(pop)

__kernel void feed_forward_cl(
        __global struct Neuron *neurons, // current layer neurons
        __global double *biases,  // biases for the current layer
        __global double *biasWeights, //weights for all the biases
        __global double *input, //weights for all the biases
        __global double *weights, // weights between current and previous layer
        __global struct Neuron *output,        // output to store calculated values
        int num_neurons,                // neurons in previous layer
        int layer_id                    // current layer ID
) {
    int id = get_global_id(0); // Get the global thread ID

    double sum = 0.0;
    // Handle input layer separately
    if (layer_id == 0) {
        output[id].value = input[id];
    } else {
        // Weighted sum computation
        for (int i = 0; i < num_neurons; i++) {
            sum += neurons[i].value * weights[id * num_neurons + i];
            if (layer_id == 2) printf("weight layer 2: %f\n", weights[id * num_neurons + i]);
        }


        // Add bias
        sum += biases[id] * biasWeights[id];

        /*// Apply activation function (ReLU in this example)
        double inter = fmax(sum, (double) 0.0);*/

        // Apply activation function (ReLU for hidden layers, softmax for output layer)
        if (layer_id != 0) {
            double inter = fmin(fmax(sum, 0.0), 1.0); // ReLU activation
        }

        // Softmax normalization for output layer only
        if (layer_id == 2) { // Output layer
            // Step 1: Compute the maximum for numerical stability
            __local double max_val;
            if (id == 0) {
                max_val = -INFINITY;
                for (int i = 0; i < num_neurons; i++) {
                    max_val = fmax(max_val, output[i]);
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE); // Synchronize threads in the work group

            // Step 2: Compute the exponentials and their sum
            __local double exp_sum;
            if (id == 0) {
                exp_sum = 0.0;
                for (int i = 0; i < num_neurons; i++) {
                    exp_sum += exp(output[i] - max_val); // Subtract max_val for numerical stability
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE); // Synchronize threads in the work group

            // Step 3: Normalize
            output[id] = exp(output[id] - max_val) / exp_sum;
        }

        // Debugging
        if (id == 152 || id == 153) {
            printf("Softmax normalized value for neuron %d: %f\n", id, output[id]);
        }

        output[id].value = fmax(sum, (double) 0.0); // ReLU activation
        if (layer_id == 2) {
            //printf("sum value layer 3: %f\n", sum);
            printf("intermediate value layer 3: %f\n", inter);
            printf("output value layer 3: %f\n", output[id].value);
        }
    }
}
