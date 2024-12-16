#pragma pack(push, 1)
struct Neuron {
    double value;
    double gradient;
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
    output[id].gradient = 0.0;

    double sum = 0.0;
    if (layer_id == 0) {
        output[id].value = input[id];
    } else {
        // Weighted sum computation
        for (int i = 0; i < num_neurons; i++) {
            sum += neurons[i].value * weights[id * num_neurons + i];
        }

        sum += biases[id] * biasWeights[id];


//        if(layer_id == 1 && sum == 0){
//            printf("neuron id")
//        }
//         output[id].value = fmax(sum, 0.0); // ReLU activation
        output[id].value = 1 / (1 + exp(-sum)); // sigmoid activation

        if(output[id].value == 0.0 && id < 1){
            for (int i = 0; i < num_neurons; i++) {
                printf("\n:neuron val: %f\n weight val: %f\n sum val: %f\n", neurons[i].value, weights[id * num_neurons + i], sum);
            }
        }
//        if(layer_id == 1){
//            printf("2nd layer: %f\n 2nd layer val: %f\n 2nd layer grad: %f\n index: %d\n",sum, output[id].value ,output[id].gradient, id);
//        }
    }
}
