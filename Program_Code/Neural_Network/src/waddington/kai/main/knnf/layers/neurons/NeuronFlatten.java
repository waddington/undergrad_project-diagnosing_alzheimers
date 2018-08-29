package waddington.kai.main.knnf.layers.neurons;

import org.jblas.DoubleMatrix;
import waddington.kai.main.knnf.LayerActivation;
import waddington.kai.main.knnf.LayerType;

import java.util.List;

/**
 * This is the class for flattening neurons. It extends the {@link Neuron} class.
 */
public class NeuronFlatten extends Neuron {

    /**
     * See {@link Neuron#Neuron(int, LayerType, LayerActivation)}.
     * @param id The neuron ID.
     */
    public NeuronFlatten(int id) {
        super(id, LayerType.flatten, null);
    }

    /**
     * Back-propagates the error from the layer above into the correct shape for the layer below.
     * @param lowerNeurons A list of neurons in the layer below.
     * @param upperNeurons A list of neurons in the layer above.
     */
    @Override
    public void calculateDeltas(List<Neuron> lowerNeurons, List<Neuron> upperNeurons) {
        // This is really slow
        // Calculate error
        double sumWeightedError = 0;
        LayerType upperType = upperNeurons.get(0).getType();

        switch (upperType) {
            case output: {
                for (int i=0; i<upperNeurons.size(); i++) {
                    sumWeightedError += upperNeurons.get(i).getError().get(0,0) * ((NeuronOutput) upperNeurons.get(i)).getWeight(getId());
                }
                break;
            }
            case fc: {
                for (int i=0; i<upperNeurons.size(); i++) {
                    sumWeightedError += upperNeurons.get(i).getError().get(0,0) * ((NeuronFullyConnected) upperNeurons.get(i)).getWeight(getId());
                }
                break;
            }
        }

        setError(new DoubleMatrix(new double[] {sumWeightedError}));

        // There are nothing to update on this layer so there are no deltas to calculate
    }

    /**
     * Doesn't do anything in a flattening layer...
     */
    @Override
    public void applyDeltas() {}
}