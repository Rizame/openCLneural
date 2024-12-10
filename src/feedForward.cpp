#pragma pack(push, 1)
struct Neuron {
    double value;
    double gradient;
};
#pragma pack(pop)

__kernel void feed_forward_cl(
        __global struct Neuron *neurons, // previous layer neurons
        __global double *biases,  // biases for the previous layer of neurons
        __global double *biasWeights, //weights for all the biases
        __global double *input, //input data
        __global double *weights, // weights between current and previous layer
        __global struct Neuron *output,        // output to store calculated values
        int num_neurons,                // neurons in previous layer
        int layer_id                    // current layer ID
) {
    int id = get_global_id(0); // Get the global thread ID

    double sum = 0.0;
    if (layer_id == 0) {
        output[id].value = input[id];
    } else {
        // Weighted sum computation
        for (int i = 0; i < num_neurons; i++) {
            sum += neurons[i].value * weights[id * num_neurons + i];
        }


        // Add bias
        sum += biases[id] * biasWeights[id];

        if(layer_id != 2) output[id].value = fmax(sum, 0.0); // ReLU activation


        if (layer_id == 2) {
            output[id].value = sum;
            printf("sum value layer 3: %f\n", sum);
        }
    }
}
