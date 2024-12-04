struct Neuron {
    double value;
    double gradient;
};

__kernel void feed_forward_cl(
        __global struct Neuron* neurons, // current layer neurons
        __global double* biases,  // biases for the current layer
        __global double* biasWeights, //weights for all the biases
        __global double* input, //weights for all the biases
        __global double* weights, // weights between current and previous layer
        __global double* output,        // output to store calculated values
        int num_neurons,                // neurons in previous layer
        int layer_id                    // current layer ID
)
{
    int id = get_global_id(0); // Get the global thread ID

    double sum = 0.0;
    // Handle input layer separately
    if (layer_id == 0) {
        output[id] = input[id];
        if(id == 152 || id == 153){
            printf("input for neuron 152-153: %f\n", input[id]);
        }
    }
    else{
        // Weighted sum computation
        for (int i = 0; i < num_neurons; i++) {
            sum += neurons[i].value * weights[id * num_neurons + i];
        }


        // Add bias
        sum += biases[id] * biasWeights[id];

        // Apply activation function (ReLU in this example)
        output[id] = fmax(sum, 0.0); // ReLU activation

        if(id == 152 || id == 153){
            printf("neuron value for neurons 152-153: %f\n", neurons[id].value);
            printf("weight value for neuron 152: %f\n", weights[id * num_neurons + 152]);
            printf("weight value for neuron 153: %f\n", weights[id * num_neurons + 153]);
            printf("calculated value for neuron 152-153: %f\n", output[id]);
        }
    }
}
