package waddington.kai.main.knnf.layers;

import org.jblas.DoubleMatrix;
import waddington.kai.main.knnf.LayerActivation;
import waddington.kai.main.knnf.LayerType;
import waddington.kai.main.knnf.NetworkHelper;
import waddington.kai.main.knnf.layers.neurons.Neuron;
import waddington.kai.main.knnf.layers.neurons.NeuronFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * This is the class for convolution layers. It extends the {@link Layer} class.
 */
public class LayerConvolution extends Layer {
    /**
     * The size of the receptive field.
     */
    private int filterSize;
    /**
     * The stride used.
     */
    private int stride;
    /**
     * The number of neurons in the layer.
     */
    private int numberOfFilters;

    /**
     * See {@link Layer#Layer(int, LayerType, LayerActivation)}.
     * @param id The layer ID.
     * @param activation The activation type for the layer.
     */
    public LayerConvolution(int id, LayerActivation activation) {
        super(id, LayerType.conv, activation);
    }

    public void setFilterSize(int filterSize) {
        this.filterSize = filterSize;
    }

    public int getFilterSize() {
        return filterSize;
    }

    public void setStride(int stride) {
        this.stride = stride;
    }

    public int getStride() {
        return stride;
    }

    public void setNumberOfFilters(int numFilters) {
        numberOfFilters = numFilters;
    }

    public int getNumberOfFilters() {
        return numberOfFilters;
    }

    /**
     * Creates the neurons for this layer, 1 neuron for each filter.
     * @param number The number of neurons to create.
     */
    @Override
    public void createNeurons(int number) {
        for (int i=0; i<numberOfFilters; i++) {
            Neuron neuron = NeuronFactory.getConvolutionNeuron(i, getInputSize(), getOutputSize(), filterSize, stride, getLayerActivation());
            addNeuron(neuron);
        }
    }

    /**
     * Sets the input data to each of the neurons in this layer.
     * @param inputNeurons The list of neurons to get the data from.
     */
    @Override
    public void setInputData(List<Neuron> inputNeurons) {
        // Get list of all input data
        List<DoubleMatrix> inputDataList = new ArrayList<>();
        for (Neuron neuron: inputNeurons)
            inputDataList.add(neuron.getOutputData());
        
        // Consolidate input data
        // Each filter uses a summed view of all input channels
        DoubleMatrix input = NetworkHelper.sumMatrices(inputDataList);

        // Set the data
        for (int i=0; i<getNumberOfNeurons(); i++) {
            getNeuron(i).setInputData(input);
        }
    }

    /**
     * Triggers the weight delta calculations for each neuron in this layer.
     * @param lowerNeurons The neurons in the layer below.
     * @param upperNeurons The neurons in the layer above.
     */
    @Override
    public void calculateDeltas(List<Neuron> lowerNeurons, List<Neuron> upperNeurons) {
        // Each neuron calculate individual errors + deltas
        for (int i=0; i<getNumberOfNeurons(); i++) {
            getNeuron(i).calculateDeltas(lowerNeurons, upperNeurons);
        }
    }

    /**
     * Triggers each neuron in this layer to apply their weight deltas.
     */
    @Override
    public void applyDeltas() {
        for (int i=0; i<getNumberOfNeurons(); i++) {
            getNeuron(i).applyDeltas();
        }
    }
}