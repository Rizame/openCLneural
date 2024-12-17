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
    std::function<double(double)> activate, activate_deriv;
    Layer(int numNeurons, int numNeuronsPrev, int layerId, std::function<double(double)> activate,
          std::function<double(double)> activateDeriv) : activate(std::move(activate)), activate_deriv(std::move(activateDeriv)) {

        neurons.resize(numNeurons); // Resize the neurons vector
        weights.resize(layerId == 0 ? 1 : numNeurons * numNeuronsPrev, 0.0);
        biasWeights.resize(layerId == 0 ? 1 : numNeurons);
        biases.resize(layerId == 0 ? 1 : numNeurons, 1.0); // Resize the biases for each neuron
        deltas.resize(layerId == 0 ? 1 : numNeurons, 0.0);
    }
};


#endif //NEURALDIGITRECON_LAYER_H
