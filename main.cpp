#include <stdio.h>
#include <iostream>
#include <vector>
#include "inc/NeuralNetwork.h"
// OpenCL includes
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

int main() {
    // Get number of OpenCL platforms
    std::vector<int> topology{784,256,10};
    NeuralNetwork nn{topology};

    return 0;
}