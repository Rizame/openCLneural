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
    std::vector<std::vector<double>> TEST_images = load_IDX3("trainData/emnist-test-images-idx3-ubyte");
    std::vector<std::vector<double>> images = load_IDX3("trainData/emnist-train-images-idx3-ubyte");

    std::vector<int> target = load_IDX1_to_array("trainData/emnist-train-labels-idx1-ubyte", images.size());
    std::vector<int> TEST_target = load_IDX1_to_array("trainData/emnist-test-labels-idx1-ubyte", TEST_images.size());

    std::vector<int> topology{784, 256, 10};

    NeuralNetwork NN{topology};


    std::vector<int> guessed{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int> TEST_guessed{0, 0};
    std::ofstream output("./src/output.txt");


    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < images.size(); i++) {
            NN.feedForward(images[i]);
            if(NN.guess == target[i]){
                guessed[j]++;
            }
            NN.backPropagate(target[i]);
            if(i % 10000 == 0) std::cout<<"| ";
        }
        double result = (double)guessed[j]/(double)images.size()*100;
        std::cout<<"Done Epoch "<<j<<std::endl;
        std::cout<<"percentage of guesses for epoch "<<j<<": "<<result<<std::endl;
    }

    for (int i = 0; i < TEST_images.size(); i++) {
        NN.feedForward(TEST_images[i]);
        if(NN.guess == TEST_target[i]){
            TEST_guessed[0]++;
        }
        NN.backPropagate(TEST_target[i]);
    }
    double test_res0 = static_cast<double>(TEST_guessed[0])/(double)TEST_images.size()*100;
    std::cout<<"percentage of guesses for TEST data1: "<<test_res0<<std::endl;

    for (int j = 4; j < 9; j++) {
        for (int i = 0; i < images.size(); i++) {
            NN.feedForward(images[i]);
            if(NN.guess == target[i]){
                guessed[j]++;
            }
            NN.backPropagate(target[i]);
            if(i % 10000 == 0) std::cout<<"| ";
        }
        double result = (double)guessed[j]/(double)images.size()*100;
        std::cout<<"Done Epoch "<<j<<std::endl;
        std::cout<<"percentage of guesses for epoch "<<j<<": "<<result<<std::endl;
    }

    for (int i = 0; i < TEST_images.size(); i++) {
        NN.feedForward(TEST_images[i]);
        if(NN.guess == TEST_target[i]){
            TEST_guessed[1]++;
        }
        NN.backPropagate(TEST_target[i]);
    }
    double test_res = static_cast<double>(TEST_guessed[1])/(double)TEST_images.size()*100;
    std::cout<<"percentage of guesses for TEST data2: "<<test_res<<std::endl;


    std::vector<double> input;
    input.resize(784);
    while(1){
        int customTarget = -1;
        std::string ans;
        std::cout<<"image process: (yes/no)"<<std::endl;
        std::cin>>ans;
        if(ans == "no") break;
        if(ans != "no" && ans != "yes") continue;
        std::cout<<"please type your actual number"<<std::endl;
        std::cin>>customTarget;

        input = NN.readCustom();
        NN.feedForward(input);
        std::cout<<"I'm guessing!: "<<NN.guess<<std::endl;
        NN.backPropagate(customTarget);
    }
    return 0;
}
