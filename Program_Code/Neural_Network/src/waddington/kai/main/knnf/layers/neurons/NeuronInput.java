package waddington.kai.main.knnf.layers.neurons;

import waddington.kai.main.knnf.LayerActivation;
import waddington.kai.main.knnf.LayerType;

import java.util.List;

/**
 * This is the class for input-layer neurons. It extends the {@link Neuron} class.
 */
public class NeuronInput extends Neuron {

    /**
     * See {@link Neuron#Neuron(int, LayerType, LayerActivation)}.
     * @param id The neuron ID.
     */
    public NeuronInput(int id) {
        super(id, LayerType.input, null);
    }

    /**
     * The input-layer neurons don't do anything here.
     * @param lowerNeurons A list of neurons in the layer below.
     * @param upperNeurons A list of neurons in the layer above.
     */
    @Override
    public void calculateDeltas(List<Neuron> lowerNeurons, List<Neuron> upperNeurons) {
        // Input layer doesn't need to do anything
    }

    /**
     * The input-layer neurons don't do anything here.
     */
    @Override
    public void applyDeltas() {
        // Input layer doesn't need to do anything
    }
}