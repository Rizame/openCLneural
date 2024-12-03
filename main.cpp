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
    std::vector<int> topology{784,256,10};
    NeuralNetwork nn{topology};


    return 0; // Success

}
