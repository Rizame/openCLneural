#include "Neuron.h"
#include "vector"

#ifndef NEURALDIGITRECON_LAYER_H
#define NEURALDIGITRECON_LAYER_H

struct Layer {
    std::vector<Neuron> neurons;
    std::vector<std::vector<double>> weights; //[current][previous]
    std::vector<double> biases;
    std::vector<double> biasWeights;
    Layer(int numNeurons, int numInputsPerNeuron, int i) {
        neurons.resize(numNeurons); // Resize the neurons vector
        weights.resize((i == 0 ? 0 : numNeurons), std::vector<double>(numInputsPerNeuron, 1.0)); // Resize weights for each neuron
        biasWeights.resize(i == 0 ? 0 : numNeurons);
        biases.resize(i == 0 ? 0 : numNeurons); // Resize the biases for each neuron

    }
};


#endif //NEURALDIGITRECON_LAYER_H
