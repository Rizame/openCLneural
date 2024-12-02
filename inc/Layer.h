#include "Neuron.h"
#include "vector"

#ifndef NEURALDIGITRECON_LAYER_H
#define NEURALDIGITRECON_LAYER_H

struct Layer {
    std::vector<Neuron> neurons;
    std::vector<std::vector<double>> weights; //[current][previous]
    std::vector<double> biases;
    Layer(int numNeurons, int numInputsPerNeuron, int i) {
        neurons.resize(numNeurons); // Resize the neurons vector
        weights.resize((i == 0 ? 0 : numNeurons), std::vector<double>(numInputsPerNeuron)); // Resize weights for each neuron
        biases.resize(i == 0 ? 0 : numNeurons); // Resize the biases for each neuron
    }
};


#endif //NEURALDIGITRECON_LAYER_H
