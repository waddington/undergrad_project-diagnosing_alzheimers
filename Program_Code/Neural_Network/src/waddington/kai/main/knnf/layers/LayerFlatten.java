package waddington.kai.main.knnf.layers;

import org.jblas.DoubleMatrix;
import waddington.kai.main.knnf.LayerActivation;
import waddington.kai.main.knnf.LayerType;
import waddington.kai.main.knnf.layers.neurons.Neuron;
import waddington.kai.main.knnf.layers.neurons.NeuronFactory;

import java.util.List;

/**
 * This is the class for the flattening layer. It extends the {@link Layer} class, and is a required layer between 2D layers and 1D layers.
 */
public class LayerFlatten extends Layer {

    /**
     * See {@link Layer#Layer(int, LayerType, LayerActivation)}.
     * @param id The layer ID.
     */
    public LayerFlatten(int id) {
        super(id, LayerType.flatten, null);
    }

    /**
     * Creates the neurons for this layer. 1 neuron per value/cell.
     * @param number The number of neurons to create.
     */
    @Override
    public void createNeurons(int number) {
        for (int i=0; i<getOutputSize()[2]; i++) {
            Neuron neuron = NeuronFactory.getFlattenNeuron(i, getInputSize(), getOutputSize());
            addNeuron(neuron);
        }
    }

    /**
     * Sets the input data to each neuron in this layer. Has to calculate the ID of the neuron in this layer as it transforms a list of 2D matrices to a 1xN matrix.
     * @param inputNeurons The list of neurons to get the data from.
     */
    @Override
    public void setInputData(List<Neuron> inputNeurons) {
        int neuronId = 0;

        for (int i=0; i<inputNeurons.size(); i++) {
            Neuron neuron = inputNeurons.get(i);
            DoubleMatrix data = neuron.getOutputData();
            int dataRows = data.rows;
            int dataCols = data.columns;

            for (int y=0; y<dataRows; y++) {
                for (int x=0; x<dataCols; x++) {
                    double cellValue = data.get(y, x);
                    // This is a vector with 1 value
                    getNeuron(neuronId).setInputData(new DoubleMatrix(new double[] {cellValue}));

                    neuronId++;
                }
            }
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
     * Doesn't do anything in a flattening layer...
     */
    @Override
    public void applyDeltas() {
        // Flattening layer doesn't need to do anything
    }
}