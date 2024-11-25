// C standard includes
#include <stdio.h>
#include <iostream>
#include <vector>
// OpenCL includes
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

int main() {
    // Get number of OpenCL platforms
    cl_uint platformCount = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &platformCount);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get OpenCL platform count: " << err << std::endl;
        return -1;
    }

    // Get all platforms
    std::vector<cl_platform_id> platforms(platformCount);
    err = clGetPlatformIDs(platformCount, platforms.data(), nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get OpenCL platforms: " << err << std::endl;
        return -1;
    }

    std::cout << "Number of OpenCL platforms: " << platformCount << std::endl;

    // Iterate through each platform and get devices
    for (cl_uint i = 0; i < platformCount; ++i) {
        char platformName[128];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platformName), platformName, nullptr);
        std::cout << "\nPlatform " << i + 1 << ": " << platformName << std::endl;

        // Get number of devices in the platform
        cl_uint deviceCount = 0;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to get OpenCL device count: " << err << std::endl;
            continue;
        }

        // Get all devices
        std::vector<cl_device_id> devices(deviceCount);
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices.data(), nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to get OpenCL devices: " << err << std::endl;
            continue;
        }

        std::cout << "Number of devices: " << deviceCount << std::endl;

        // Print device details
        for (cl_uint j = 0; j < deviceCount; ++j) {
            char deviceName[128];
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
            std::cout << "  Device " << j + 1 << ": " << deviceName << std::endl;
        }
    }

    return 0;
}