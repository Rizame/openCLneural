#pragma pack(push, 1)
struct Neuron {
    double value; // Neuron value after activation
    double gradient; // For storing temporary derivatives if needed
};
#pragma pack(pop)

__kernel void back_propagation_cl(__global struct Neuron *prevLayerNeurons, // Previous layer neurons
                                  __global struct Neuron *currentLayerNeurons, // Current layer neurons
                                  __global double *weights, // Weights connecting prev layer to current layer
                                  __global double *weightsNext, // Weights connecting current layer to next layer
                                  __global double *weightUpdates, // Buffer to store updated weights
                                  __global double *nextLayerDeltas, // Deltas of the next layer
                                  __global double *deltas, // Deltas for the current layer
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
        double newWeight = oldWeight - learningRate * inputValue * delta;
        weightUpdates[id * numPrevLayerNeurons + i] = newWeight; // Save updated weight
    }

}
