#include <stdio.h>
#include <iostream>
#include <vector>

#include "inc/NeuralNetwork.h"

#include "inc/input_parse.h"

// OpenCL includes
#ifdef __APPLE__

#include <OpenCL/cl.h>
#else

#include <CL/cl.h>

#endif

int main() {
    std::vector<std::vector<double>> images = load_IDX3("trainData/train-images.idx3-ubyte");
    std::vector<int> target = load_IDX1_to_array("trainData/train-labels.idx1-ubyte", images.size());
    std::vector<int> topology{784, 256, 10};
    NeuralNetwork NN{topology};
//    NN.feedForward(images[0]);
//    NN.errorCalculation(target[0]);
//    NN.backPropagate(target[0]);
//    NN.feedForward(images[0]);
//    NN.errorCalculation(target[0]);

    int guessed = 0;
    for(int i = 0; i < 2500; i++){
        std::cout<<"Feeding: "<<target[i]<<std::endl;
        NN.feedForward(images[i]);
        if(NN.guess == target[i]) guessed++;
        NN.errorCalculation(target[i]);
        NN.backPropagate(target[i]);
    }
    std::cout<<"Guess%: "<<(float)guessed/2500<<std::endl;

    return 0; // Success

}
