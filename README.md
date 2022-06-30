# AIM-Intel-Blogathon
Blogathon conducted by AIM and Intel

Intel® FPGA Add-on for oneAPI Base Toolkit — Hands on Introduction to Deep Learning Inference with Intel® FPGAs
Continuing from my previous three blogs, this time I will introduce you to the Field-programmable gate arrays (FPGAs) and its use in deep learning inferences.

My previous three blogs:

Hands on Guide to Intel® oneAPI HPC Toolkit | by Tamal Acharya | Apr, 2022 | Medium

Hands On Guide to Intel® AI Analytics Toolkit | by Tamal Acharya | Apr, 2022 | Medium

Intel® Distribution of OpenVINO™ toolkit — — Optimised Deep Learning | by Tamal Acharya | Mar, 2022 | Medium

Field-programmable gate arrays (FPGAs) are powerful, programmable hardware accelerators (offload and inline) that can be configured into custom solutions to speed up key workloads, process data from custom sources (streams), perform low-latency computation, and many more use-cases. The Intel® FPGA Add-on for oneAPI Base Toolkit is a specialized component for programming these reconfigurable devices. When paired with the Intel® oneAPI DPC++/C++ Compiler, the FPGA add-on allows developers to compile an FPGA bitstream, configuring these flexible platforms to meet a broad range of application needs.

Few good resources for learning FPGA and its applications:

· Intel® FPGAs for Artificial Intelligence (AI)

· Intel® FPGA Academic Program

· Intel® FPGAs and Programmable Devices-Intel® FPGA

· Blogs — Intel Communities


(Source: Developer Kits for IoT (intel.com))

Prerequisites
Please follow and install the necessary software before continuing.

The following resources are encouraged for developers who are new to DPC++ or FPGA architecture.

Essentials of SYCL

Introduction to SYCL
SYCL Program Structure
SYCL FPGA Optimization Guide

(Source: FPGA Development Flow Using Intel® oneAPI Base Toolkit and Intel® FPGA Add-On for oneAPI Base Toolkit: Program FPGAs Faster)

Software Requirements
64-bit Linux* Software Development Environment with g++

Intel® Distribution of OpenVINO™ toolkit Linux for FPGA version R3

Hardware Requirements
Intel® Programmable Acceleration Card with Intel® Arria® 10 FPGA GX FPGA

(Code credits for the below hands on exercises: Intel AI Academy: Deep Learning Inference using Intel FPGA)

Hands on 1: Intel® Distribution of OpenVINO™ toolkit Application for the CPU
In this hands on we will be practicing writing an OpenVINO toolkit inference engine application.

The Intel® Deep Learning Deployment Toolkit that’s part of the Intel® Distribution of OpenVINO™ toolkit allows you to easily accelerate and deploy CNNs on Intel® platforms including CPUs, GPUs, VPUs, and FPGAs.

In this hands on we will practice writing a simple cat classification application using the OpenVINO toolkit. The code written can be applied to any of the supported Intel® platforms but for ease of deployment we will target the CPU first and then the FPGA in the next exercise.

This application takes video or images as input and will classify the object in the image as one of 32 types of dogs or cats.

Step 1. Setup Lab Environment
____ 1. Open a terminal in your Linux* system

____ 2. Extract the exercise files associated with the training

a. “tar -xf cat_dog_classification.tar.gz”

____ 3. Change directory (cd) into the extracted directory, this directory will now be refered to as <Lab Dir>

____ 4. Examine the environment script init_openvino.sh


Figure 1. Environment script init_openvino.sh

In the script there are two sections, OpenVINO environment and Intel® FPGA envirnoment settings.

____ 5. Edit the script to match your environment

a. Make sure IE_INSTALL is set to the location of the installed deployment tools

b. If you’re using one of the Programmable Acceleration Cards, ensure to run the init_env.sh in the script

c. Set the INTELFPGAOCLSDKROOT to the Intel FPGA OpenCL™ runtime

d. Set AOCL_BOARD_PACKAGE_ROOT to the opencl_bsp directory with in the Acceleration Stack directory.

____ 6. Source init_opencl.sh from the terminal

a. source init_openvino.sh

____ 7. Create a build directory in the <Lab Dir> and cd into it

a. mkdir build

b. cd build

c. cmake ..

This will create the build environment.

____ 8. Build the sample application

a. make all

____ 9. Test the application

a. Type: ../bin/intel64/Release/demo -h

You should see the options for running the demo as shown in the following figure.


Figure 2. Screen capture of the options for running the demo

Step 2. Write the Inference Engine Code for the Application
The Cat and Dog classification application we’re going to be writing will run an image through a provided deep learning network. The network is the output of the Model Optimizer provided in an Intermediate Representation (IR) format (.xml and .bin). We’re going to be completing the code for the following inference process. Should you encounter any issues you may also consult the solution directory.


Figure 3. Inference setup and workflow

____ 1. Change directory into the source code directory

a. cd <Lab Dir>/demo

____ 2. Open main.cpp in your favorite text editor

____ 3. Look for the comment TODO: Step 1, Load the Plugin

The first step in the inference process is to load the appropriate plugin. The OpenVINO API can be used with a variety of devices so you can easily switch between inference accelerators with the same network and source code. Different plugins support different accelerator devices.

____ 4. Assign the variable plugin to the plugin gotten from the device flag FLAGS_d passed in from the command line

plugin = dispatcher.getPluginByDevice(FLAGS_d);

The device defaults to CPU, which is what we’ll be using in this exercise, in exercise 2, we will attempt to run the inference on the FPGA using the -d device flag.

____ 5. Look for the comment TODO: Step 2, Read IR Generated by Model Optimizer

The next step in the inference process is to read the IR generated by the optimizer. The IR is passed in by the -m flag and is already stored in the variable modelPath.

The function LoadNework is already written for you (in network.cpp) which makes it easy to load in both the .xml topology file and .bin weights file. The LoadNetwork function also performs the function of configuring and allocating input and output structures of the inference.

____ 6. Complete the code to load the network below the Step 3 comment

network -> LoadNetwork(modelPath, 1);

The arguments for the function are the path to the .xml IR file and the batch size.

____ 7. Look for the comment TODO: Step 4 Load Model

This step will load the read model into the plugin.

____ 8. Below the comment, complete the code using the already written LoadNetworkToPlugin() function

Network->LoadNetworkToPlugin();

____ 9. Look for while loop few lines below.

If the input is a video or a camera feed, this while loop will loop through the frames and perform inference on each one. If the source is an image then it will perform just one inference.

____ 10. Locate Step 6 Prepare Input

The ConvertBGR function is already written for you (in util.cpp) to convert the RGB information from the frame and convert it to the BGR format that most networks are trained with. While the function is converting, it also assigns the value to the Input data structure needed prior to performing inference.

____ 11. Locate the comment “TODO: Step 7 Perform Inference”

We are now ready to perform inference on the frame.

____ 12. Right below the comment, write the code to perform the inference

Network->Infer();

The function written for you (in network.cpp) will call the Infer function on the plugin and perform error checks.

____ 13. Locate “Step 8 Process Output”

This is the final step in the inference process. This function will parse the outputs to look for the top N results, after which the application will display the frame along with the classification performance in infers/sec, the top inference results and associated confidence value.

____ 14. Save the file main.cpp

____ 15. Back in the terminal, go to the build directory

a. cd <Lab Dir>/build

____ 16. Type “make” to build the application

Fix any compile errors that may have risen.

____ 17. Go to location of the compiled executable

a. cd <Lab Dir>/bin/intel64/Release/

____ 18. Execute the application with one of the provided images

____ 19. ./demo -i cat6.jpg -l labels.txt -m breed_fp32.xml

The exercise includes several dog and cat images, choose the image you like and pass it to the application with the -i flag.

Use the -l flag to specify the labels.txt which contains the named labels for the 32 different dog and cat classes.

The -m flag is used to specify the intermediate representation. The exercise includes two IRs for single precision and half precision floating point data types. For execution on the CPU use the 32 bit IR. The intermediate representation was generated from a Tensorflow* model using the Model Optimizer.


Figure 4. Cat classification results

You should see something similar to the above. Notice the performance as executed on the CPU.

Hands on 2: Perform Inference on an FPGA
In this hands on we will continue with the application from hands on 1. You have to complete hands on 1 in order for you to proceed with hands on 2.

Here we will be practicing executing the same topology on various different FPGA images and compare performance results

Step 1. Setup FPGA Lab Environment
____ 10. Open a terminal in your Linux system

____ 11. Go to the <Lab Dir> from Exercise 1

____ 12. Examine the environment script init_openvino.sh


Figure 1. Environment script init_openvino.sh

In the script there are two sections, OpenVINO™ toolkit environment and Intel® FPGA envirnoment settings.

____ 13. Edit the script if necessary to make sure all the FPGA paths are correct

____ 14. Source init_opencl.sh from the terminal if you haven’t already done so

a. source init_openvino.sh

____ 15. Run “aocl diagnose”, you should see an FPGA board connected


Figure 2. Screen capture showing FPGA board is connected

____ 16. Go to the directory where all the DLA FPGA images are located

a. cd $IE_INSTALL/../a10_dcp_bitstreams

If you’re not using an Intel® Programmable Acceleration Card with Intel® Arria® 10 GX FPGA, go to the appropriate bitstream directory.

____ 17. Examine the contents of the directory

You should see many bitstreams optimized for different topologies and data types.

____ 18. Load the Generic FP16 bitstream

a. aocl program acl0 2–0–1_RC_FP16_Generic.aocx

Ensure programming was successful.

____ 19. Perform Diagnostics by running “aocl diagnose acl0”

Ensure you see the DIAGNOSTIC_PASSED message.

Step 2. Perform Inference with the FPGA
____ 20. Change directory into the <Lab Dir>/bin/intel64/Release

____ 21. Execute inference with the CPU

./demo -i dog2.jpg -l labels.txt -m breed_fp32.xml -d CPU

Notice the performance as well as the result.

____ 22. Perform the same inference with the FPGA using the 32bit network

./demo -i dog2.jpg -l labels.txt -m breed_fp32.xml -d HETERO:FPGA,CPU

Notice the confidence results, should expect slightly different result since the FPGA is doing the operations in FP16.

Also notice the performance.

Use the HETERO plug in to fall back to the CPU whenever a primitive is not supported on the FPGA. Here our network has a softmax layer that must be executed on the CPU.

____ 23. Perform the inference with the FPGA using the 16bit network

./demo -i dog2.jpg -l labels.txt -m breed_fp16.xml -d HETERO:FPGA,CPU

Notice the performance again. You should not see a difference in performance since the FPGA plugin simply truncates the values and the actual calculations are still done in FP16 on the FPGA as before.

____ 24. Lets try to run a more optimized 16bit FPGA image

a. pushd .

b. cd $IE_INSTALL/../a10_dcp_bitstreams

c. aocl program acl0 2–0–1_RC_FP16_GoogleNet.aocx

d. popd

Because our classification network is based on the GoogLeNet topology, we can use a more optimized FPGA image, removing the primitives that GoogLeNet doesn’t need, allowing more Processing Elements to be placed onto the FPGA to accelerate convolutions.

____ 25. Perform the inference with the FPGA again

./demo -i dog2.jpg -l labels.txt -m breed_fp16.xml -d HETERO:FPGA,CPU

You should now see a significant increase in performance.

____ 26. FPGAs can also be configured with lesser data types for better performance, let’s try to run the inference using FP11 Generic image

a. pushd .

b. cd $IE_INSTALL/../a10_dcp_bitstreams

c. aocl program acl0 2–0–1_RC_FP11_Generic.aocx

d. popd

____ 27. Run the inference again

./demo -i dog2.jpg -l labels.txt -m breed_fp16.xml -d HETERO:FPGA,CPU

You should see an performance increase again.

____ 28. Lets try the FP11 FPGA image optimized for GoogLeNet

a. pushd .

b. cd $IE_INSTALL/../a10_dcp_bitstreams

c. aocl program acl0 2–0–1_RC_FP11_GoogleNet.aocx

d. popd

____ 29. Run the inference again

./demo -i dog2.jpg -l labels.txt -m breed_fp16.xml -d HETERO:FPGA,CPU

As you can see due to the flexible nature of FPGAs, you’ll see improvements in performance as you tailor the FPGA image to your network or by using lesser data types.


Figure 3. Dog classification results

Intel® oneAPI DPC++ FPGA Optimization Guide
If you want to check out FPGA application for DPC+ check this out.

Use this guide to learn about:

Introduction To FPGA Design Concepts: Describes FPGA design concepts.
Analyze Your Design: Describes how to work with FPGA optimization report and Intel® VTune Profiler.
Optimize Your Design: Describes how to achieve high performance by optimizing the throughput and use various resources.
FPGA Optimization Flags, Attributes, Pragmas, and Extensions: Describes a list of compiler optimization flags, attributes, pragma, and extensions that allow you to customize the kernel compilation process.
Quick Reference: A cheat sheet of all FPGA-specific attributes, pragmas, and variables.
Additional References:

Intel® FPGA Add-on for oneAPI Base Toolkit Resources

Intel® oneAPI DPC++ FPGA Optimization Guide
Explore DPC++ Through Intel® FPGA Code Samples
Intel® FPGA Add-on for oneAPI Base Toolkit Training Modules

Accelerate FPGA Programming
FPGA Development Flow Using Intel® oneAPI Base Toolkit
https://software.seek.intel.com/oneapievents
Intel® FPGA Add-on for oneAPI Base Toolkit
Articles References

https://tinyurl.com/vh2etv79
#oneAPI
