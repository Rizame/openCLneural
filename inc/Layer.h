#include <functional>
#include <utility>
#include <utility>
#include "Neuron.h"
#include "vector"

#ifndef NEURALDIGITRECON_LAYER_H
#define NEURALDIGITRECON_LAYER_H

struct Layer {
    std::vector<Neuron> neurons;
    std::vector<double> deltas;
    std::vector<double> weights;
    std::vector<double> biases;
    std::vector<double> biasWeights;
    Layer(int numNeurons, int numNeuronsPrev, int layerId) {

        neurons.resize(numNeurons); // Resize the neurons vector
        weights.resize(layerId == 0 ? 0 : numNeurons * numNeuronsPrev, 0.0);
        biasWeights.resize(layerId == 0 ? 0 : numNeurons);
        biases.resize(layerId == 0 ? 0 : numNeurons, 1.0); // Resize the biases for each neuron
        deltas.resize(layerId == 0 ? 0 : numNeurons, 0.0);
    }
};


#endif //NEURALDIGITRECON_LAYER_H
