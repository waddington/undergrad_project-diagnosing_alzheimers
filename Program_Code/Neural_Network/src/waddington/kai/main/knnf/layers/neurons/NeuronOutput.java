package waddington.kai.main.knnf.layers.neurons;

import org.jblas.DoubleMatrix;
import waddington.kai.main.knnf.LayerActivation;
import waddington.kai.main.knnf.LayerType;
import waddington.kai.main.knnf.NetworkManager;

import java.util.List;

/**
 * This is the class for output-layer neurons. It extends the {@link Neuron} class.
 */
public class NeuronOutput extends Neuron {
    /**
     * A {@link DoubleMatrix} containing the weights for this neuron (excluding weight to bias value).
     */
    private DoubleMatrix weights;
    /**
     * The bias value.
     */
    private double bias;
    /**
     * The delta for the bias.
     */
    private double biasDelta;
    /**
     * The expected classification for the current example.
     */
    private int expectedOutput;

    /**
     * See {@link Neuron#Neuron(int, LayerType, LayerActivation)}.
     * @param id The neuron ID.
     * @param activation The {@link LayerActivation} for this neuron.
     */
    public NeuronOutput(int id, LayerActivation activation) {
        super(id, LayerType.output, activation);
    }

    /**
     * Initialises the values for the weights. The min/max values are dictated by {@link NetworkManager#MaximumInitialWeights} and are generated used {@link NetworkManager#random}.
     */
    public void initWeights(int numberOfWeights) {
        weights = new DoubleMatrix(numberOfWeights);

        for (int i=0; i<numberOfWeights; i++) {
            double value = NetworkManager.random.nextDouble() * NetworkManager.MaximumInitialWeights;
            weights.put(i, value);
        }

        bias = NetworkManager.random.nextDouble() * NetworkManager.MaximumInitialWeights;
        setDeltas(DoubleMatrix.zeros(weights.length));
    }

    /**
     * Sets the weights for this neuron to an already existing set of weights.
     * @param w A {@link DoubleMatrix} containing the weights.
     */
    public void setWeights(DoubleMatrix w) {
        weights = w;
    }

    /**
     * Get the weights for this neuron.
     * @return The weights this neuron uses.
     */
    public DoubleMatrix getWeights() {
        return weights;
    }

    /**
     * Get a weight by its index.
     * @param index The index of the weight to retrieve.
     * @return The value of the weight.
     */
    public double getWeight(int index) {
        return weights.get(index, 0);
    }

    public void setBias(double b) {
        bias = b;
    }

    public double getBias() {
        return bias;
    }

    /**
     * Sets the input data for the neuron and triggers the neuron to calculate its output.
     * @param data A {@link DoubleMatrix} containing the input data.
     */
    @Override
    public void setInputData(DoubleMatrix data) {
        super.setInputData(data);

        calculateOutput();
    }

    /**
     * Calculates the neurons output. This is an element-wise multiplication between the neurons weights and inputs.
     */
    private void calculateOutput() {
        double output = weights.dot(getInputData()) + bias;

        setOutputData(new DoubleMatrix(new double[] {output}));
    }

    /**
     * Sets the expected classification output and also calculates the squared error of this neuron so that the MSE can be calculated. The output data at this point has already had the softmax function applied.
     * @param expectedOutput The expected classification of the current example.
     * @return The squared error of this neuron.
     */
    public double setExpectedOutput(int expectedOutput) {
        this.expectedOutput = expectedOutput;

        double squaredError = Math.pow(expectedOutput - getOutputData().get(0,0), 2);

        return squaredError;
    }

    /**
     * Calculates the weight and bias deltas.
     * @param lowerNeurons A list of neurons in the layer below.
     * @param upperNeurons A list of neurons in the layer above.
     */
    @Override
    public void calculateDeltas(List<Neuron> lowerNeurons, List<Neuron> upperNeurons) {
        // Calculate error
        double error  = expectedOutput - getOutputData().get(0,0);
        setError(new DoubleMatrix(new double[] {error}));
        
        // Ensure deltas matrix exists
        if (getDeltas() == null) {
            setDeltas(DoubleMatrix.zeros(weights.length));
        }

        // Calculate deltas
        DoubleMatrix deltasMomentum = getDeltas().mul(NetworkManager.Momentum);
        DoubleMatrix deltas = getInputData().mul(error);
        deltas = deltas.mul(NetworkManager.LearningRate);
        deltas.addi(deltasMomentum);

        double biasMomentum = NetworkManager.Momentum * biasDelta;
        biasDelta = (NetworkManager.LearningRate * bias * error) + biasMomentum;

        setDeltas(deltas);
    }

    /**
     * Applies the weight and bias deltas.
     */
    @Override
    public void applyDeltas() {
        bias += biasDelta;
        weights.addi(getDeltas());
    }
}