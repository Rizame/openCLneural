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
//    std::vector<std::vector<double>> testinput;
//    testinput.push_back({0.4,0.6});
//    testinput.push_back({0.2,0.5,0.6});
//    testinput.push_back({0.9,0.6,0.7});
//    testinput.push_back({0.1,0.0,0.8});


//    NeuralNetwork test{std::vector<int>{2,2,1}};
//
//    test.feedForward(testinput[0]);
//    test.backPropagate(2);

    for (int j = 0; j < 100; j++) {
        for (int i = 0; i < 400; i++) {
            NN.feedForward(images[i]);
            NN.backPropagate(target[i]);
        }
        NN.feedForward(images[0]);
        NN.errorCalculation(target[0]);
    }

//    NN.feedForward(images[0]);
//    NN.errorCalculation(target[0]);
//    NN.backPropagate(target[0]);
//    NN.feedForward(images[0]);
//    NN.errorCalculation(target[0]);

//    int guessed = 0;
//    for(int i = 0; i < 2500; i++){
//        std::cout<<"Feeding: "<<target[i]<<std::endl;
//        NN.feedForward(images[i]);
//        if(NN.guess == target[i]) guessed++;
//        NN.errorCalculation(target[i]);
//        NN.backPropagate(target[i]);
//    }
//    std::cout<<"Guess%: "<<(float)guessed/2500<<std::endl;
//
//    return 0; // Success

}
