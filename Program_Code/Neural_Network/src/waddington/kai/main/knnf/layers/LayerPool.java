package waddington.kai.main.knnf.layers;

import waddington.kai.main.knnf.LayerActivation;
import waddington.kai.main.knnf.LayerType;
import waddington.kai.main.knnf.layers.neurons.Neuron;
import waddington.kai.main.knnf.layers.neurons.NeuronFactory;

import java.util.List;

/**
 * This is the class for the pooling layers. It extends the {@link Layer} class.
 */
public class LayerPool extends Layer {
    /**
     * The size of the receptive field.
     */
    private int poolSize;
    /**
     * The stride used.
     */
    private int stride;

    /**
     * See {@link Layer#Layer(int, LayerType, LayerActivation)}.
     * @param id The layer ID.
     */
    public LayerPool(int id, LayerType type) {
        super(id, type, null);
    }

    public void setPoolSize(int poolSize) {
        this.poolSize = poolSize;
    }

    public int getPoolSize() {
        return poolSize;
    }

    public void setStride(int stride) {
        this.stride = stride;
    }

    public int getStride() {
        return stride;
    }

    /**
     * Creates the neurons for this layer, 1 neuron for each input channel.
     * @param number The number of neurons to create.
     */
    @Override
    public void createNeurons(int number) {
        for (int i=0; i<number; i++) {
            Neuron neuron = NeuronFactory.getPoolNeuron(i, getInputSize(), getOutputSize(), getLayerType(), poolSize, stride);
            addNeuron(neuron);
        }
    }

    /**
     * Sets the input data to each of the neurons in this layer.
     * @param inputNeurons The list of neurons to get the data from.
     */
    @Override
    public void setInputData(List<Neuron> inputNeurons) {
        for (int i=0; i<getNumberOfNeurons(); i++) {
            getNeuron(i).setInputData(inputNeurons.get(i).getOutputData());
        }
    }

    /**
     * Triggers the weight delta calculations for each neuron in this layer.
     * @param lowerNeurons The neurons in the layer below.
     * @param upperNeurons The neurons in the layer above.
     */
    @Override
    public void calculateDeltas(List<Neuron> lowerNeurons, List<Neuron> upperNeurons) {
        for (int i=0; i<getNumberOfNeurons(); i++) {
            getNeuron(i).calculateDeltas(lowerNeurons, upperNeurons);
        }
    }

    /**
     * Doesn't do anything in a pooling layer...
     */
    @Override
    public void applyDeltas() {
        // Pooling layer doesn't need to do anything
    }
}