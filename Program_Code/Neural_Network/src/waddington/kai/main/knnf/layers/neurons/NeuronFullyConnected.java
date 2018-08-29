package waddington.kai.main.knnf.layers.neurons;

import org.jblas.DoubleMatrix;
import waddington.kai.main.knnf.LayerActivation;
import waddington.kai.main.knnf.LayerType;
import waddington.kai.main.knnf.NetworkHelper;
import waddington.kai.main.knnf.NetworkManager;

import java.util.List;

/**
 * This is the class for fully-connected neurons. It extends the {@link Neuron} class.
 */
public class NeuronFullyConnected extends Neuron {
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
     * See {@link Neuron#Neuron(int, LayerType, LayerActivation)}.
     * @param id The neuron ID.
     * @param activation The {@link LayerActivation} for this neuron.
     */
    public NeuronFullyConnected(int id, LayerActivation activation) {
        super(id, LayerType.fc, activation);
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
        output = NetworkHelper.applyActivation(getActivation(), output);

        setOutputData(new DoubleMatrix(new double[] {output}));
    }

    /**
     * Calculates the weight and bias deltas.
     * @param lowerNeurons A list of neurons in the layer below.
     * @param upperNeurons A list of neurons in the layer above.
     */
    @Override
    public void calculateDeltas(List<Neuron> lowerNeurons, List<Neuron> upperNeurons) {
        // Calculate error
        double error = calculateError(upperNeurons);
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
     * Back-propagates the error from the layer above to know how responsible this neuron is for the error in the network.
     * @param upperNeurons A list of neurons in the layer above.
     */
    private double calculateError(List<Neuron> upperNeurons) {
        double outputDerivative = NetworkHelper.applyActivationDerivative(getActivation(), getOutputData().get(0,0));
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

        double error = sumWeightedError * outputDerivative;

        return error;
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