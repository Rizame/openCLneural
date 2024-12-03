struct Neuron {
    double value;
    double gradient;
};

__kernel void feed_forward_cl(
        __global const Neuron* neurons, //current layer neurons
        __global const double* biases,
        __global const double* weights,
        __global const float* input, // image data
        __global double* output, // output to store calculated values
        int num_neurons; //neurons in orevious layer
        int layer_id;
) {
    int id = get_global_id(0);// Get the global thread ID (which is unique for each thread)
    double sum = 0.0;
    for(int i = 0; i < num_neurons, i++){
        sum += neurons[i].value * weights[id][i];
    }
    //activation func
    output[id] = sum;

    // Initialize biases to 0
}