package waddington.kai.main;

import waddington.kai.main.knnf.NetworkManager;
import waddington.kai.main.knnf.serialisation.NetworkLoader;

public class Main {
	/**
	 * The instance of a KNNF network.
	 */
    private NetworkManager network;

    // This method calls the necessary methods to load an existing network

	/**
	 * Used to load an existing network model
	 * @param filename The name (without extension) of the file of an existing model in the "LoadNetwork" folder.
	 */
	private void loadNetwork(String filename) {
        System.out.println("Loading network...");
        network = NetworkLoader.loadNetwork(filename);
    }

	/**
	 * Sets up a new network.
	 * <p>
	 * Sets all of the parameters of a network as well as creating the network architecture.
	 */
	private void createNetwork() {
        network = new NetworkManager();

        // Hyperparameters
        network.setLearningRate(0.0001f);
        network.setMaximumInitialWeights(0.0001f);
        network.setMomentum(0.001f);

        // Add layers to network
        System.out.println("Building network...");
        
        network.addInput(256, 256, 1);
        network.addPool(2, 2, "max");

        network.addConv(64, 9, 1, "lrelu");
        network.addConv(128, 9, 1, "lrelu");
        network.addPool(2, 2, "max");

        network.addConv(64, 9, 1, "lrelu");
        network.addConv(128, 9, 1, "lrelu");
        network.addPool(2, 2, "max");

        network.addConv(64, 9, 1, "lrelu");
        network.addConv(128, 5, 1, "lrelu");
        network.addPool(2, 2, "max");

        network.addFlatten();
        network.addFC(1024, "tanh");
        network.addOutput(3);
    }

	/**
	 * Triggers a validity check of the network. Required before network is used.
	 */
	private void networkValidityCheck() {
        network.networkValidityCheck();
    }

	/**
	 * Starts the network training.
	 * <p>
	 * Handles setting the paths to the training data.
	 */
	private void trainNetwork() {
        network.setTrainingData("./../mri-png/");
        // network.setTerminationConditions(-1, -1, (long) (1000*60*60*11));
        network.setTerminationConditions(1, -1, -1); // 1 epoch
        network.startTraining("./../mri-png/");
    }

	/**
	 * Starts the network testing.
	 * <p>
	 * Handles setting the paths to the testing data.
	 */
    private void testNetwork() {
        network.setTestingData("./../mri-png/");
        network.startTesting("./../mri-png/");
    }

	/**
	 * Starts the network predicting.
	 * <p>
	 * Handles setting the paths to the prediction data.
	 */
    private void getPrediction() {
        network.setPredictionData("./../mri-png/");
        network.startPredicting("./../mri-png/");
    }

    public static void main(String[] args) {
        Main main = new Main();
        main.createNetwork();
        // main.loadNetwork("Network-1525630996414");
        main.networkValidityCheck();
        main.trainNetwork();
        // main.testNetwork();
        // main.getPrediction();
    }
}
