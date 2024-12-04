#pragma pack(push, 1)
struct Neuron {
    double value;
    double gradient;
};
#pragma pack(pop)

__kernel void feed_forward_cl(
        __global struct Neuron* neurons, // current layer neurons
        __global double* biases,  // biases for the current layer
        __global double* biasWeights, //weights for all the biases
        __global double* input, //weights for all the biases
        __global double* weights, // weights between current and previous layer
        __global struct Neuron* output,        // output to store calculated values
        int num_neurons,                // neurons in previous layer
        int layer_id                    // current layer ID
)
{
    int id = get_global_id(0); // Get the global thread ID

    double sum = 0.0;
    // Handle input layer separately
    if (layer_id == 0) {
        output[id].value = input[id];
    }
    else{
        // Weighted sum computation
        for (int i = 0; i < num_neurons; i++) {
            sum += neurons[i].value * weights[id * num_neurons + i];
            if(layer_id == 2) printf("weight layer 2: %f\n", weights[id * num_neurons + i]);
        }


        // Add bias
        sum += biases[id] * biasWeights[id];

        // Apply activation function (ReLU in this example)
        double inter = fmax(sum, 0.0);
        output[id].value = fmax(sum, 0.0); // ReLU activation
        if(layer_id == 2){
            //printf("sum value layer 3: %f\n", sum);
            printf("intermediate value layer 3: %f\n", inter);
            printf("output value layer 3: %f\n", output[id].value);
        }
    }
}
