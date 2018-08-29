# Diagnosing Alzheimers Disease

This repo contains the project that I completed as part of the final year of my computer science undergraduate course. While I still plan to develope this, I will do so in other repo's to preserve the state of this at the time of submission.

## This project was:

* A neural network framework built from scratch using Java. I would consider the code for this project to be a framework as it is not tied to any particular network architecture, and it is very easy to create new architectures using this project.
* An application of this framework to try and diagnose Alzheimers Disease from just brain MRI images using a convolutional neural network.

## Continuations...

1. I will add a link here in the near future to the repo that I use for further development of the neural network framework.
1. I will add a link here in the near future to the repo that I use for further developing a neural network model to predict Alzheimers Disease from just brain MRI images. I will not be using this framework for that, as there are other frameworks that are much more optimised than this will ever be.

## Notes

1. The `./out/` directory must exist in the project root.
1. The `./error-logging/` directory must exist in the project root.
1. The following `.jar` files must be located in a `./libraries/` directory in project root.
	* `commons-io-2.6.jar`
	* `google-guava.jar`
	* `hamcrest-core-1.3.jar`
	* `jblas-1.2.4.jar`
	* `json-simple-1.1.jar`
	* `junit-4.12.jar`
	* `zip4j_1.3.2.jar`

## Compiling/Running
1. Run the `compile.sh` bash script located in the project root.
2. After compiling, run the `run.sh` bash script located in the project root.
* Alternatively run the `compileRun.sh` bash script to compile and run the program.
* Run the `unitTest.sh` bash script to run all of the unit tests.

## To Use
* Your program will only need to access the `NetworkManager` class.
* Your program must set up a few things for the network:
    1. You must either load a network or create one.
    2. You must call `networkValidityCheck()` before using the network.

## To create a network
* You must call `setLearningRate()`, `setMaximumInitialWeights()`, and `setMomentum()` to set the network hyperparameters. This must be done before adding any layers to the network.
* Adding layers:
    1. Adding an input layer: `addInput()` - you must pass the dimensions of the input data in the form of [Y, X, Z].
    1. Adding convolutional layers: `addConv()` - you must pass the number of filters, size of the filters, stride, and the activation type.
    1. Adding pooling layers: `addPool()` - you must pass the pool size, stride, and pool type (max / min).
    1. Adding a flattening layer: `addFlatten()` - you must call this layer once, and before the first fully connected or output layer. This layer transitions from the convolutional/pooling layers to the fully connected/output layers.
    1. Adding a fully connected layer: `addFC()` - you must pass the number of neurons in this layer and the activation type.
    1. Add the output layer: `addOutput()` - you must pass the number of output nodes required.

## Activation types
* Convolutional, and fully connected layers have the choice of the following activation types:
    1. Linear
    1. Sigmoid
    1. Tanh
    1. Relu
    1. Leaky Relu
* The output layer uses the softmax activation function.

## Training the network
Before training the network you must specify training termination conditions. You can choose whether to train for a specific number of epochs, a certain amount of time, or until the MSE falls below a certain value. Use `-1` to not use a specific termination condition. At least one termination condition must be set.

## Network serialisation
The network will be saved after training has terminated, it will be saved in a `.zip` file in the project root. For large networks the saving process can be slow, and results in a file that might be a few hundred MB's. There will be the option to save the network every so often.