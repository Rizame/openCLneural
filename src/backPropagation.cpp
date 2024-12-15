#pragma pack(push, 1)
struct Neuron {
    double value; // Neuron value after activation
    double gradient; // For storing temporary derivatives if needed
};
#pragma pack(pop)

__kernel void back_propagation_cl(__global struct Neuron *prevLayerNeurons, // Previous layer neurons
                                  __global struct Neuron *currentLayerNeurons, // Current layer neurons
                                  __global double *weights, // Weights connecting prev layer to current layer
                                  __global double *weightUpdates, // Buffer to store updated weights
                                  __global double *nextLayerDeltas, // Deltas of the next layer
                                  __global double *deltas, // Deltas for the current layer
                                  int numPrevLayerNeurons, // Number of neurons in previous layer
                                  int numCurrentLayerNeurons, // Number of neurons in current layer
                                  double learningRate, // Learning rate
                                  int isOutputLayer, // 1 if this is the output layer, 0 otherwise
                                  int targetIndex // Target index for classification (only used in output layer)
) {
    int j = get_global_id(0); // Each thread handles one neuron in the current layer
    if (j >= numCurrentLayerNeurons) return;

    double delta = 0.0;
    double value = currentLayerNeurons[j].value;

    // Compute delta for output layer
    if (isOutputLayer) {
        // Target value is 1 if targetIndex == j, otherwise 0
        double targetValue = (j == targetIndex) ? 1.0 : 0.0;
        delta = (value - targetValue) * (value * (1.0 - value)); // Derivative of sigmoid
    } else {
        // Compute delta for hidden layer
        double sum = 0.0;
        for (int l = 0; l < numCurrentLayerNeurons; l++) {
            double weight = weights[l * numPrevLayerNeurons + j];
            sum += weight * nextLayerDeltas[l];
        }
        delta = sum * (value * (1.0 - value)); // Derivative of sigmoid
    }

    // Store the computed delta for the current neuron
    deltas[j] = delta;

    // Update weights
    for (int i = 0; i < numPrevLayerNeurons; i++) {
        double oldWeight = weights[j * numPrevLayerNeurons + i];
        double inputValue = prevLayerNeurons[i].value;
        double newWeight = oldWeight - learningRate * inputValue * delta;
        weightUpdates[j * numPrevLayerNeurons + i] = newWeight; // Save updated weight
    }
}
