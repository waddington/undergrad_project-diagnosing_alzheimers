package waddington.kai.main.knnf.layers;

import org.jblas.DoubleMatrix;
import waddington.kai.main.knnf.LayerActivation;
import waddington.kai.main.knnf.LayerType;
import waddington.kai.main.knnf.NetworkHelper;
import waddington.kai.main.knnf.layers.neurons.Neuron;
import waddington.kai.main.knnf.layers.neurons.NeuronFactory;

import java.util.List;

/**
 * This is the class for fully-connected layers. It extends the {@link Layer} class.
 */
public class LayerFullyConnected extends Layer {
    /**
     * The number of neurons in the layer.
     */
    private int numberOfNeurons;
    /**
     * The number of weights that each neuron will have (excluding weight to bias).
     */
    private int numberOfWeights;

    /**
     * See {@link Layer#Layer(int, LayerType, LayerActivation)}.
     * @param id The layer ID.
     * @param activation The activation type for the layer.
     */
    public LayerFullyConnected(int id, LayerActivation activation) {
        super(id, LayerType.fc, activation);
    }

    public void setNumberOfNeurons(int numberOfNeurons) {
        this.numberOfNeurons = numberOfNeurons;
    }

    public int getNumberOfNeurons() {
        return numberOfNeurons;
    }

    public void setNumberOfWeights(int number) {
        numberOfWeights = number;
    }

    public int getNumberOfWeights() {
        return numberOfWeights;
    }

    /**
     * Creates the neurons for this layer.
     * @param number The number of neurons to create.
     */
    @Override
    public void createNeurons(int number) {
        for (int i=0; i<number; i++) {
            Neuron neuron = NeuronFactory.getFCNeuron(i, getLayerActivation(), numberOfWeights);
            addNeuron(neuron);
        }
    }

    /**
     * Sets the input data to each neuron. Joins values from all inputs to a 1xN matrix.
     * @param inputNeurons The list of neurons to get the data from.
     */
    @Override
    public void setInputData(List<Neuron> inputNeurons) {
        // Merge input data
        DoubleMatrix joinedInputs = NetworkHelper.joinInputVectors(inputNeurons);

        // Set the input data for each neuron
        for (int i=0; i<getNumberOfNeurons(); i++) {
            getNeuron(i).setInputData(joinedInputs);
        }
    }

    /**
     *  Triggers the weight delta calculations for each neuron in this layer.
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
     * Triggers each neuron in this layer to apply their weight deltas.
     */
    @Override
    public void applyDeltas() {
        for (int i=0; i<getNumberOfNeurons(); i++) {
            getNeuron(i).applyDeltas();
        }
    }
}