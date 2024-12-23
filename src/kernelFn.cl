#pragma pack(push, 1)
struct Neuron {
    double value;
};
#pragma pack(pop)

__kernel void feed_forward(__global struct Neuron *neurons, // previous layer neurons
                           __global double *biasWeights,    // weights for biases
                           __global double *weights,        // weights between current and previous layer
                           int prev_neurons,                // number of neurons in previous layer
                           int layer_id                     // current layer ID
) {


    int id = get_global_id(0); // Get the global thread ID
    double sum = 0.0;
    int topology[] = {784,256,10};

    int all_prev_neurons = 0;
    int weight_offset = 0;
    for (int i = 0; i < layer_id; i++) {
        all_prev_neurons += topology[i];
        weight_offset += topology[i] * topology[i + 1];
    }


    for (int i = 0; i < prev_neurons; i++) {
        sum += neurons[all_prev_neurons + i].value * weights[weight_offset + id * prev_neurons + i];
        printf("\n:weight id: %d",weight_offset + id * prev_neurons + i );
    }

    sum += biasWeights[ all_prev_neurons - topology[0] + id];


    if (weights[id * prev_neurons + 0] < -10000000) printf("\nGG");
    printf("\n:sum val: %f", sum);
    neurons[prev_neurons+id].value = 1 / (1 + exp(-sum)); // sigmoid activation
    printf("\n:neuron val: %f",neurons[prev_neurons+id].value);
//    if (neurons[prev_neurons+id].value == 0.0 && id < 1) {
//        for (int i = 0; i < prev_neurons; i++) {
//            printf("\n:neuron val: %f\n weight val: %f\n sum val: %f\n", neurons[i].value,
//                   weights[id * prev_neurons + i], sum);
//        }
//    }

}

__kernel void init(
        __global double* weights,        // Buffer to store weights
        __global double* biases,         // Buffer to store biases
        __global double* seeds,
        int num_weights,           // Total number of weights
        int num_biases          // Total number of biases
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

__kernel void back_propagation(__global struct Neuron *prevLayerNeurons, // Previous layer neurons
                                  __global struct Neuron *currentLayerNeurons, // Current layer neurons
                                  __global double *weights, // Weights connecting prev layer to current layer
                                  __global double *weightsNext, // Weights connecting current layer to next layer
                                  __global double *nextLayerDeltas, // Deltas of the next layer
                                  __global double *deltas, // Deltas for the current layer
                                  __global double *biasWeights,
                                  int numPrevLayerNeurons, // Number of neurons in previous layer
                                  int numCurrentLayerNeurons, // Number of neurons in current layer
                                  int numNextLayerNeurons, // Number of neurons in next layer
                                  double learningRate, // Learning rate
                                  int isOutputLayer, // 1 if this is the output layer, 0 otherwise
                                  int targetIndex // Target index for classification (only used in output layer)
) {
    int id = get_global_id(0); // Each thread handles one neuron in the current layer
    if (id >= numCurrentLayerNeurons) return;

    double delta = 0.0;
    double value = currentLayerNeurons[id].value;

    // Compute delta for output layer
    if (isOutputLayer) {

        double targetValue = (id == targetIndex) ? 1.0 : 0.0;
        delta = (value - targetValue); // Derivative of sigmoid

    }
    else {
        // Compute delta for hidden layer
        double sum = 0.0;
        for (int l = 0; l < numNextLayerNeurons; l++) {
            double weight = weightsNext[l * numCurrentLayerNeurons + id];
            sum += weight * nextLayerDeltas[l];
        }
        delta = sum; // Derivative of sigmoid
    }

    // Store the computed delta for the current neuron
    deltas[id] = delta * (value * (1.0 - value));


    // Update weights
    for (int i = 0; i < numPrevLayerNeurons; i++) {
        double oldWeight = weights[id * numPrevLayerNeurons + i];
        double inputValue = prevLayerNeurons[i].value;
        weights[id * numPrevLayerNeurons + i] = oldWeight - learningRate * inputValue * delta;
    }
    double oldBiasWeight = biasWeights[id];
    biasWeights[id] = oldBiasWeight - learningRate * deltas[id];

}
