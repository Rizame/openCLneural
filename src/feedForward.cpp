#pragma pack(push, 1)
struct Neuron {
    double value;
};
#pragma pack(pop)

__kernel void feed_forward_cl(__global struct Neuron *neurons, // previous layer neurons
                              __global double *biases,  // biases for the previous layer of neurons
                              __global double *biasWeights, //weights for all the biases
                              __global double *input, //input data
                              __global double *weights, // weights between current and previous layer
                              __global struct Neuron *output,        // output to store calculated values
                              int num_neurons,                // neurons in previous layer
                              int layer_id                    // current layer ID
) {
    int id = get_global_id(0); // Get the global thread ID
    output[id].value = 0.0;

    double sum = 0.0;
    if (layer_id == 0) {
        output[id].value = input[id];
    } else {
        // Weighted sum computation
        for (int i = 0; i < num_neurons; i++) {
            sum += neurons[i].value * weights[id * num_neurons + i];
        }

        sum += biases[id] * biasWeights[id];


        if (weights[id * num_neurons + 0] < -10000000) printf("\nGG");

        output[id].value = 1 / (1 + exp(-sum)); // sigmoid activation

        if (output[id].value == 0.0 && id < 1) {
            for (int i = 0; i < num_neurons; i++) {
                printf("\n:neuron val: %f\n weight val: %f\n sum val: %f\n", neurons[i].value,
                       weights[id * num_neurons + i], sum);
            }
        }

    }
}
