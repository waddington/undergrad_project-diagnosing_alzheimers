package waddington.kai.main.knnf.layers;

import waddington.kai.main.knnf.LayerActivation;
import waddington.kai.main.knnf.LayerType;
import waddington.kai.main.knnf.layers.neurons.Neuron;

import java.util.ArrayList;
import java.util.List;

/**
 * This is the abstract class that all layer-type-specific classes extend.
 */
public abstract class Layer {
    /**
     * Each layer has an ID and is it's position in the network.
     */
    private int id;
    /**
     * The specific type that the layer is.
     */
    private LayerType type;
    /**
     * The activation type that the layer uses. Sometimes null.
     */
    private LayerActivation activation;

    // Data shapes
    private int[] inputSize;
    private int[] outputSize;

    /**
     * The neurons in this layer.
     */
    private List<Neuron> neurons;

    /**
     * This is the only constructor that can be used. Sets the ID, layer type, and activation type of the layer.
     * @param id The ID for the layer.
     * @param type The {@link LayerType} that the new layer is.
     * @param activation The {@link LayerActivation} that the layer will use.
     */
    public Layer(int id, LayerType type, LayerActivation activation) {
        this.id = id;
        this.type = type;
        this.activation = activation;
    }

    /**
     * Get the ID of the layer.
     * @return The ID of the layer.
     */
    public int getId() {
        return id;
    }

    /**
     * Get the type of the layer.
     * @return The type of the layer.
     */
    public LayerType getLayerType() {
        return type;
    }

    /**
     * Get the activation type that the layer uses.
     * @return The activation type that the layer uses.
     */
    public LayerActivation getLayerActivation() {
        return activation;
    }

    /**
     * Set the ID of the layer.
     * @param i The ID to use.
     */
    public void setId(int i) {
        id = i;
    }

    /**
     * Set the type of the layer.
     * @param t The type to use.
     */
    public void setLayerType(LayerType t) {
        type = t;
    }

    /**
     * Set the activation type of the layer.
     * @param a The activation type to use.
     */
    public void setLayerActivation(LayerActivation a) {
        activation = a;
    }

    public void setInputSize(int[] size) {
        inputSize = size;
    }

    public int[] getInputSize() {
        return inputSize;
    }

    public void setOutoutSize(int[] size) {
        outputSize = size;
    }

    public int[] getOutputSize() {
        return outputSize;
    }

    /**
     * Used to create all of the neurons in the layer. Is abstract so all individual layer types must have their own implementation for this.
     * @param number The number of neurons to create.
     */
    public abstract void createNeurons(int number);

    /**
     * Adds an instance of {@link Neuron} to this layer.
     * @param neuron The neuron to add.
     */
    public void addNeuron(Neuron neuron) {
        if (neurons == null)
            neurons = new ArrayList<>();

        neurons.add(neuron);
    }

    /**
     * Gets the list of neurons in this layer.
     * @return The list of neurons in this layer.
     */
    public List<Neuron> getNeurons() {
        return neurons;
    }

    /**
     * Gets a specific neuron by ID from this layer.
     * @param id The ID of the neuron to get.
     * @return The neuron.
     */
    public Neuron getNeuron(int id) {
        return neurons.get(id);
    }

    /**
     * Gets the number of neurons in this layer.
     * @return The number of neurons in this layer.
     */
    public int getNumberOfNeurons() {
        return neurons.size();
    }

    /**
     * Sets the input data for the neurons in a layer.
     * @param inputNeurons The list of neurons to get the data from.
     */
    public abstract void setInputData(List<Neuron> inputNeurons);

    /**
     * Calculates the weight deltas for a given list of neurons.
     * @param lowerNeurons The neurons in the layer below.
     * @param upperNeurons The neurons in the layer above.
     */
    public abstract void calculateDeltas(List<Neuron> lowerNeurons, List<Neuron> upperNeurons);

    /**
     * Applies the weight deltas to the neurons.
     */
    public abstract void applyDeltas();
}