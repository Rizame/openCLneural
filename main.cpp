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
    std::vector<std::vector<double>> TEST_images = load_IDX3("trainData/t10k-images.idx3-ubyte");
    std::vector<std::vector<double>> images = load_IDX3("trainData/train-images.idx3-ubyte");

    std::vector<int> target = load_IDX1_to_array("trainData/train-labels.idx1-ubyte", images.size());
    std::vector<int> TEST_target = load_IDX1_to_array("trainData/t10k-labels.idx1-ubyte", TEST_images.size());

 std::vector<int> topology{784, 256, 10};
//    std::vector<int> topology{3, 3, 3};
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

    for (int i = 0; i < 10; i++) {
        NN.feedForward(images[0]);
        //NN.errorCalculation(5);
        //NN.backPropagate(5);
    }

//    std::vector<int> guessed{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
//    std::vector<int> TEST_guessed{0, 0};
//
//    for (int j = 0; j < 5; j++) {
//        for (int i = 0; i < images.size(); i++) {
//            NN.feedForward(images[i]);
//            if(NN.guess == target[i]){
//                guessed[j]++;
//            }
//            NN.backPropagate(target[i]);
//            if(i % 10000 == 0) std::cout<<"| ";
//        }
//        std::cout<<"Done Epoch "<<j<<std::endl;
//        NN.feedForward(images[0]);
//        NN.errorCalculation(target[0]);
//    }
//
//    for (int i = 0; i < TEST_images.size(); i++) {
//        NN.feedForward(TEST_images[i]);
//        if(NN.guess == TEST_target[i]){
//            TEST_guessed[0]++;
//        }
//    }
//    double test_res0 = static_cast<double>(TEST_guessed[0])/60000*100;
//    std::cout<<"percentage of guesses for TEST data1: "<<test_res0<<std::endl;
//
//    for (int j = 5; j < 10; j++) {
//        for (int i = 0; i < images.size(); i++) {
//            NN.feedForward(images[i]);
//            if(NN.guess == target[i]){
//                guessed[j]++;
//            }
//            NN.backPropagate(target[i]);
//            if(i % 10000 == 0) std::cout<<"| ";
//        }
//        std::cout<<"Done Epoch "<<j<<std::endl;
//        NN.feedForward(images[0]);
//        NN.errorCalculation(target[0]);
//    }
//
//    for (int i = 0; i < TEST_images.size(); i++) {
//        NN.feedForward(TEST_images[i]);
//        if(NN.guess == TEST_target[i]){
//            TEST_guessed[1]++;
//        }
//    }
//    double test_res = static_cast<double>(TEST_guessed[1])/60000*100;
//    std::cout<<"percentage of guesses for TEST data2: "<<test_res<<std::endl;
//
//    for(int i = 0; i < 10; i++){
//        double result = (double)guessed[i]/60000*100;
//        std::cout<<"percentage of guesses for epoch "<<i<<": "<<result<<std::endl;
//    }
//
    return 0;
//
}
