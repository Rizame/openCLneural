#pragma pack(push, 1)
struct Neuron {
    double value;
};
#pragma pack(pop)

__kernel void feed_forward(
        __global struct Neuron *neurons,     // previous layer neurons
        __global double *biasWeights,       // weights for biases
        __global double *weights,           // weights between layers
        __global int *topology,            // topology of the network
        int layer_id                       // current layer ID

) {
    int id = get_global_id(0); // Get the global thread ID
    if (id >= topology[layer_id]) return; // Ensure we stay within current layer's neurons

    double sum = 0.0;
    int prev_neurons = topology[layer_id - 1];

    // Compute neuron_offset for the previous layer
    int neuron_offset_prev = 0;
    for (int i = 0; i < layer_id; i++) {
        neuron_offset_prev += topology[i];
    }

    // Compute weight_offset for the current layer
    int weight_offset = 0;
    for (int i = 0; i < layer_id; i++) {
        weight_offset += topology[i - 1] * topology[i];
    }


    if (weight_offset >= 204000) {
        printf("[ERROR] Invalid weight_offset: %d for Layer %d\n", weight_offset, layer_id);
        return;
    }

    // Compute contributions from the previous layer
    for (int i = 0; i < prev_neurons; i++) {
        int weight_index = weight_offset + id * prev_neurons + i;

        // Bounds check for weight_index
        if (weight_index >= 204000) {
            printf("[ERROR] Out-of-bounds Weight Index: %d, Layer %d, Neuron ID %d\n",
                   weight_index, layer_id, id);
            return;
        }

        // Accumulate contribution
        sum += neurons[neuron_offset_prev - prev_neurons + i].value * weights[weight_index];

        // Debugging output
        /*printf("[DEBUG] Layer %d, Neuron %d, Prev Neuron %d, Weight Index: %d, Weight: %f, Contribution: %f\n",
               layer_id, id, i, weight_index, weights[weight_index],
               neurons[neuron_offset_prev + i].value * weights[weight_index]);*/
    }

    // Add the bias term
    int bias_index = neuron_offset_prev + id - topology[0];
    if (bias_index >= 204000 || bias_index < 0) {
        printf("[ERROR] Invalid Bias Index: %d for Layer %d, Neuron ID %d\n", bias_index, layer_id, id);
        return;
    }

    sum += biasWeights[bias_index];

    //if(layer_id == 2)printf("[DEBUG] Layer %d, Neuron %d, sum: %f\n",layer_id, id, sum);
    // Apply activation function
    neurons[neuron_offset_prev + id].value = 1 / (1 + exp(-sum));
    //printf("\n just inserted into neuron id:  %d",neuron_offset_prev + id );
    // Debugging output for neuron value
    //printf("[DEBUG] Layer %d, Neuron %d, Value (After Activation): %f\n", layer_id, neuron_offset_prev +  id, neurons[neuron_offset_prev + id].value);

//    if(id == 255 && layer_id == 1){
//        printf("\nlatest weight id %d",weight_offset + id * prev_neurons + prev_neurons );
//        printf("\n neuron id: %d, neuron_offset_prev:%d",neuron_offset_prev + id, neuron_offset_prev);
//        printf("\n neuron val: %f",neurons[neuron_offset_prev + id].value);
//    }
//    if(id == 9 && layer_id == 2){
//        printf("\nlatest id %d",weight_offset + id * prev_neurons + prev_neurons );
//        printf("\n neuron id: %d",neuron_offset_prev + id);
//        printf("\n neuron val: %f",neurons[neuron_offset_prev + id].value);
//    }

}


__kernel void init(
        __global double *weights,        // Buffer to store weights
        __global double *biases,         // Buffer to store biases
        __global double *seeds,
        int num_weights,           // Total number of weights
        int num_biases          // Total number of biases
) {
    int id = get_global_id(0);       // Get the global thread ID (which is unique for each thread)

    if (id < num_weights) {

        weights[id] = sin((id + seeds[id]) * 5.1928667898);  // Random number between -1 and 1 using sine
        weights[id] = fmod(weights[id], 1.0); // Normalize between 0 and 1


        if (weights[id] < -1) {
            weights[id] = -1.0;
        }
        if (weights[id] > 1) {
            weights[id] = 1.0;
        }

    }
    if (id < num_biases) {
        biases[id] = sin((id + seeds[id]) * 112.74932);  // Random number between -1 and 1 using sine
        biases[id] = fmod(biases[id], (double) 1.0);  // Normalize between 0 and 1

        if (biases[id] < -1) {
            biases[id] = -1.0;
        }
        if (biases[id] > 1) {
            biases[id] = 1.0;
        }
    }
    // Initialize biases to 0
}

__kernel void back_propagation(__global struct Neuron *neurons,
                               __global double *weights, // Weights connecting prev layer to current layer
                               __global double *deltas, // Deltas for the current layer
                               __global double *biasWeights,
                               __global int *topology,
                               int isOutputLayer, // 1 if this is the output layer, 0 otherwise
                               int targetIndex, // Target index for classification (only used in output layer)
                               int layer_id,
                               double learningRate // Learning rate
) {
    int id = get_global_id(0); // Each thread handles one neuron in the current layer
    if (id >= topology[layer_id] || layer_id == 0) return;
    int number_of_layers = 3; //TODO make as argument

    //784, 256, 10
    int neuron_offset = 0;
    int updated_w_offset = 0;

    int weight_offset_current = 0;
    int weight_offset_prev = 0;
    int weight_offset_next = 0;


    for (int i = 0; i < layer_id; i++) {
        weight_offset_prev = weight_offset_current; // Store the current offset as the previous offset
        neuron_offset += topology[i];
        if (i - 2 >= 0) updated_w_offset += topology[i - 2];
        // Only add if `i + 1` is within bounds
        if (i - 1 >= 0) {
            weight_offset_current += topology[i] * topology[i - 1]; // Add weights for the current layer
        }
        if (i + 1 < number_of_layers) {
            weight_offset_next += topology[i] * topology[i + 1]; // Add weights for the next layer
        }
    }


    double delta = 0.0;
    double value = neurons[neuron_offset + id].value;


    // Compute delta for output layer
    if (isOutputLayer) {

        double targetValue = (id == targetIndex);
        delta = (value - targetValue); // Derivative of sigmoid

    } else {
        // Compute delta for hidden layer
        double sum = 0.0;
        int deltas_offset = neuron_offset;
        for (int i = 1; i < topology[layer_id + 1] + 1; i++) {
            double weight = weights[weight_offset_next + id * topology[layer_id] + i - 1];
            if (deltas_offset + i - 1 >= deltas_offset && deltas_offset + i - 1 < deltas_offset+256)
            sum += weight * deltas[deltas_offset + i - 1];
            else
                printf("[ERROR] Out of bounds DELTAS %d, the start: %d, end: %d", deltas_offset + i - 1, deltas_offset, deltas_offset+256 );
        }
        delta = sum;

    }

    // Store the computed delta for the current neuron
    deltas[neuron_offset + id] = delta * (value * (1.0 - value)); ///done


    // Update weights
    for (int i = 0; i < topology[layer_id - 1] + 1; i++) {
        double oldWeight = weights[weight_offset_prev + id * i + i];
        double inputValue = neurons[updated_w_offset + i].value;

        weights[id * i + i - 1 + weight_offset_prev] = oldWeight - learningRate * inputValue * delta;
    }
    double oldBiasWeight = biasWeights[id];
    biasWeights[id] = oldBiasWeight - learningRate * deltas[id];

}