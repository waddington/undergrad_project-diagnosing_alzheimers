package waddington.kai.main.knnf.layers.neurons;

import waddington.kai.main.knnf.LayerActivation;
import waddington.kai.main.knnf.LayerType;

/**
 * Static factory class for creating new {@link Neuron} instances.
 */
public class NeuronFactory {

    /**
     * Creates and sets up an instance of {@link NeuronInput}.
     * @param id The ID for the neuron.
     * @param inputSize The shape of the input data to the neuron.
     * @param outputSize The shape of the output data from the neuron.
     * @return An instance of {@link NeuronInput}.
     */
    public static Neuron getInputNeuron(int id, int[] inputSize, int[] outputSize) {
        NeuronInput neuron = new NeuronInput(id);
        neuron.initMemory(inputSize, outputSize);

        return neuron;
    }

    /**
     *  Creates and sets up an instance of {@link NeuronConvolution}.
     * @param id The ID for the neuron.
	 * @param inputSize The shape of the input data to the neuron.
	 * @param outputSize The shape of the output data from the neuron.
     * @param filterSize The size of the receptive field that the neuron will use.
     * @param stride The stride that the neuron will use.
     * @param activation The {@link LayerActivation} type that this neuron will use.
     * @return An instance of {@link NeuronConvolution}.
     */
    public static Neuron getConvolutionNeuron(int id, int[] inputSize, int[] outputSize, int filterSize, int stride, LayerActivation activation) {
        NeuronConvolution neuron = new NeuronConvolution(id, filterSize, stride, activation);
        neuron.initMemory(inputSize, outputSize);
        neuron.initFilter();

        return neuron;
    }

    /**
     *  Creates and sets up an instance of {@link NeuronPool}.
     * @param id The ID for the neuron.
	 * @param inputSize The shape of the input data to the neuron.
	 * @param outputSize The shape of the output data from the neuron.
     * @param type The type of pooling operation, see {@link LayerType}.
     * @param poolSize The size of the receptive field that the neuron will use.
     * @param stride The stride that the neuron will use.
     * @return An instance of {@link NeuronPool}.
     */
    public static Neuron getPoolNeuron(int id, int[] inputSize, int[] outputSize, LayerType type, int poolSize, int stride) {
        NeuronPool neuron = new NeuronPool(id, type);
        neuron.initMemory(inputSize, outputSize);
        neuron.setPoolSize(poolSize);
        neuron.setStride(stride);

        return neuron;
    }

    /**
     *  Creates and sets up an instance of {@link NeuronFlatten}.
     * @param id The ID for the neuron.
	 * @param inputSize The shape of the input data to the neuron.
	 * @param outputSize The shape of the output data from the neuron.
     * @return An instance of {@link NeuronFlatten}.
     */
    public static Neuron getFlattenNeuron(int id, int[] inputSize, int[] outputSize) {
        NeuronFlatten neuron = new NeuronFlatten(id);
        int[] dataSize = new int[] {1, 1, 1};
        neuron.initMemory(dataSize, dataSize);

        return neuron;
    }

    /**
     *  Creates and sets up an instance of {@link NeuronFullyConnected}.
     * @param id The ID for the neuron.
     * @param activation The {@link LayerActivation} type that this neuron will use.
     * @param numberOfWeights The number of weights that this neuron will have (excluding weight to bias).
     * @return An instance of {@link NeuronFullyConnected}.
     */
    public static Neuron getFCNeuron(int id, LayerActivation activation, int numberOfWeights) {
        NeuronFullyConnected neuron = new NeuronFullyConnected(id, activation);
        int[] dataSize = new int[] {1, 1, 1};
        neuron.initMemory(dataSize, dataSize);
        neuron.initWeights(numberOfWeights);

        return neuron;
    }

    /**
     *  Creates and sets up an instance of {@link NeuronOutput}.
     * @param id The ID for the neuron.
     * @param activation The {@link LayerActivation} type that this neuron will use.
     * @param numberOfWeights The number of weights that this neuron will have (excluding weight to bias).
     * @return An instance of {@link NeuronOutput}.
     */
    public static Neuron getOutputNeuron(int id, LayerActivation activation, int numberOfWeights) {
        NeuronOutput neuron = new NeuronOutput(id, activation);
        int[] dataSize = new int[] {1, 1, 1};
        neuron.initMemory(dataSize, dataSize);
        neuron.initWeights(numberOfWeights);

        return neuron;
    }
}