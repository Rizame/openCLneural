struct Neuron {
    double value;
    double gradient;
};

__kernel void feed_forward_cl(
        __global const Neuron* neurons, // current layer neurons
        __global const double* biases,  // biases for the current layer
        __global const double* weights, // weights between current and previous layer
        __global double* output,        // output to store calculated values
        int num_neurons,                // neurons in previous layer
        int layer_id                    // current layer ID
)
{
    int id = get_global_id(0); // Get the global thread ID
    if (id >= num_neurons) {
    return; // Prevent out-of-bounds access
    }

    double sum = 0.0;

    // Handle input layer separately
    if (layer_id == 0) {
    output[id] = input[id];
    return;
    }

    // Weighted sum computation
    for (int i = 0; i < num_neurons; i++) {
    sum += neurons[i].value * weights[id * num_neurons + i];
    }

    // Add bias
    sum += biases[id];

    // Apply activation function (ReLU in this example)
    output[id] = fmax(sum, 0.0); // ReLU activation
}
