package waddington.kai.main.knnf;

import org.jblas.DoubleMatrix;
import waddington.kai.main.knnf.exceptions.InvalidLayerOrderException;
import waddington.kai.main.knnf.exceptions.InvalidNetworkException;
import waddington.kai.main.knnf.exceptions.InvalidOutputSizeException;
import waddington.kai.main.knnf.exceptions.MissingTerminationConditionException;
import waddington.kai.main.knnf.layers.Layer;
import waddington.kai.main.knnf.layers.LayerFactory;
import waddington.kai.main.knnf.layers.LayerInput;
import waddington.kai.main.knnf.layers.LayerOutput;
import waddington.kai.main.knnf.layers.neurons.Neuron;
import waddington.kai.main.knnf.serialisation.NetworkSaver;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.*;

/**
 * The facade to the KNNF library/package.
 * This class serves as a way for a user to use the neural network package simply.
 * This class also orchestrates the training/testing loops.
 */
public class NetworkManager {
    // Hyper-parameters
	/**
	 * The learning rate of the network model.
	 */
    public static float LearningRate;
	/**
	 * The maximum distance from zero creating the range that weights can be initialised within.
	 */
	public static float MaximumInitialWeights;
	/**
	 * The momentum of the network model.
	 */
    public static float Momentum;
	/**
	 * Package-wide Random instance.
	 */
	public static Random random;

    private List<String[]> trainingCsvData;
    private int trainingExampleCount;

    private List<String[]> testingCsvData;
    private int testingExampleCount;

    private List<String[]> predictionCsvData;
    private int predictionExampleCount;

    /**
     * Stores all of the layers of the network.
     */
    private static List<Layer> networkLayers;

    // Termination conditions
    /**
     * Option to terminate training after a number of epochs.
     * One of the 3 termination conditions must be set before training.
     */
    private int terminationEpoch;
    /**
     * Option to terminate training when the MSE falls below this value.
     * One of the 3 termination conditions must be set before training.
     */
    private int terminationError;
    /**
     * Option to terminate training after a duration of time.
     * One of the 3 termination conditions must be set before training.
     */
    private long terminationDuration;

    /**
     * A flag for whether the current network is in a valid state for use.
     */
    private boolean networkValid;
    private PrintWriter errorLogger;

    private String networkReference;

    /**
     * Instantiates many settings required for the network.
     */
    public NetworkManager() {
        random = new Random(13081996); // Ensures consistency when recreating networks
        networkLayers = new ArrayList<>();

        terminationEpoch = -1;
        terminationError = -1;
        terminationDuration = -1;

        networkValid = false;

        trainingCsvData = new ArrayList<>();
        testingCsvData = new ArrayList<>();
        predictionCsvData = new ArrayList<>();

        // Used as a name for log files etc
        networkReference = String.valueOf(System.currentTimeMillis());
    }

    /**
     * Sets the learning rate of the network.
     * @param learningRate The value for the learning rate.
     */
    public void setLearningRate(float learningRate) {
        NetworkManager.LearningRate = learningRate;
    }

    /**
     * Sets the maximum distance from zero creating the range that weights can be initialised within.
     * If set to 0.1, then the range that weights can be is [-0.1, 0.1].
     * @param maxInitialWeights The absolute value for the range of weights.
     */
    public void setMaximumInitialWeights(float maxInitialWeights) {
        NetworkManager.MaximumInitialWeights = maxInitialWeights;
    }

    /**
     * Sets the momentum of the network.
     * @param momentum The value for the momentum.
     */
    public void setMomentum(float momentum) {
        NetworkManager.Momentum = momentum;
    }

    // Methods to add layers
    // All use a factory class.

    /**
     * Public method to add the input layer to a network.
     * Uses {@link LayerFactory} to create the layer.
     * @param imgWidth The width of the input images.
     * @param imgHeight The height of the input images.
     * @param imgChannels The number of channels in the input images.
     */
    public void addInput(int imgWidth, int imgHeight, int imgChannels) {
        addLayer(LayerFactory.getInputLayer(imgWidth, imgHeight, imgChannels));
    }

    /**
     * Public method to add a convolution layer to a network.
     * Uses {@link LayerFactory} to create the layer.
     * @param numFilters The number of filters the convolution layer should contain.
     * @param filterSize The size of the receptive field for all filters in the layer. This sets both the X and Y dimensions.
     * @param stride The stride that the convolution layer should use.
     * @param activation The {@link LayerActivation} that the layer should use.
     */
    public void addConv(int numFilters, int filterSize, int stride, String activation) {
        addLayer(LayerFactory.getConvLayer(numFilters, filterSize, stride, activation, getNumberOfNeuronsToLayer(networkLayers.size()-1)));
    }

    /**
     * Public method to add a pooling layer to a network.
     * Uses {@link LayerFactory} to create the layer.
     * @param poolSize The size of the receptive field for the pooling layer. This sets both the X and Y dimensions.
     * @param stride The stride that the pool layer should use.
     * @param poolType The type of pool operation. {@link LayerType#minPool} or {@link LayerType#maxPool}.
     */
    public void addPool(int poolSize, int stride, String poolType) {
        addLayer(LayerFactory.getPoolLayer(poolSize, stride, poolType, getNumberOfNeuronsToLayer(networkLayers.size()-1)));

    }

    /**
     * Public method to add a flattening layer to a network.
     * This is a required layer between multidimensional layers and single dimension layers. Must come before the first fully-connected layer.
     * Uses {@link LayerFactory} to create the layer.
     */
    public void addFlatten() {
        addLayer(LayerFactory.getFlattenLayer(getNumberOfNeuronsToLayer(networkLayers.size()-1)));
    }

    /**
     * Public method to add a fully-connected layer to a network.
     * Uses {@link LayerFactory} to create the layer.
     * @param numberOfNeurons The number of neurons that the fully-connected layer should contain.
     * @param activation The activation type - {@link LayerActivation} - that this layer should use.
     */
    public void addFC(int numberOfNeurons, String activation) {
        addLayer(LayerFactory.getFCLayer(numberOfNeurons, activation, getNumberOfNeuronsToLayer(networkLayers.size()-1)));
    }

    /**
     * Public method to add the output layer to a network.
     * Uses {@link LayerFactory} to create the layer.
     * @param numberOfOutputs The number of output classes for the data-set. This defines the number of neurons in the layer.
     */
    public void addOutput(int numberOfOutputs) {
        addLayer(LayerFactory.getOutputLayer(numberOfOutputs, getNumberOfNeuronsToLayer(networkLayers.size()-1)));
    }

    /**
     * Method used to add a {@link Layer} instance to the network.
     * The public methods to add a layer to the network call this method passing a Layer instance that {@link LayerFactory} provides. This adds the layer to the list of layers in the network ({@link NetworkManager#networkLayers}).
     * @param layer A {@link Layer} instance to add to the network.
     */
    private void addLayer(Layer layer) {
        networkLayers.add(layer);
    }

    /**
     * Method used to set the layers of a network to an already existing list of layers.
     * @param layers A list containing instances of a {@link Layer} class.
     */
    public void setLayers(List<Layer> layers) {
        networkLayers = layers;
    }

    /**
     * Gets the shape of the input to a specified layer.
     * @param layerId The ID of the layer that you wish to find the input shape for.
     * @return An int[] containing the {Z,Y,X} dimensions of the input data to the specified layer.
     */
    public static int[] getLayerInputSize(int layerId) {
        if (layerId > networkLayers.size())
            return new int[3];
            
        return networkLayers.get(layerId-1).getOutputSize();
    }

    /**
     * Used to get the list of layers in the network.
     * @return The list of layers in the network - {@link #networkLayers}.
     */
    public List<Layer> getLayers() {
        return networkLayers;
    }

    /**
     * Gets the number of neurons in the layer before the specified layer.
     * @param layerId The layer that you wish to find the number of neurons leading into.
     * @return The number of neurons in the layer before the specified layer.
     */
    public static int getNumberOfNeuronsToLayer(int layerId) {
        int number = 1;
        
        if (layerId > 0)
            number = networkLayers.get(layerId).getNumberOfNeurons();

        return number;
    }

    /**
     * Checks that the network is valid.
     * This method must be called before training/testing/predicting. It checks that the layers are in a valid order and that the data sizes are also valid, however, it shouldn't be possible to reach this stage if the data sizes are not valid.
     */
    public void networkValidityCheck() {
        System.out.println("Checking network validity...");

        checkLayerOrders();
        checkDataSizes();

        networkValid = true;
    }

    /**
     * Checks that the layers in the network are in a valid order. Called by {@link #networkValidityCheck()}. Uses the {@link LayerOrder} data structure.
     * This loops through all of the layers in the network and checks that all the layers directly before each layer are valid precedents to that layer. This also checks that the first and last layers are input and output layers.
     */
    private void checkLayerOrders() {
        Map<LayerType, LayerOrder> orders = NetworkHelper.createLayerOrderMap();

        for (int i=0; i<networkLayers.size(); i++) {
            Layer layer = networkLayers.get(i);
            LayerType type = layer.getLayerType();
            LayerOrder order = orders.get(type);

            // Check first layer is input layer
            if (i == 0)
                if (type != LayerType.input)
                    throw new InvalidLayerOrderException("\r\nFirst layer in network must be of type \"LayerType.input\".");

            // Check last layer is output layer
            if (i == networkLayers.size()-1)
                if (type != LayerType.output)
                    throw new InvalidLayerOrderException("\r\nLast layer in network must  be of type \"LayerType.output\".");

            // Check all layers' antecedents
            if (i > 0) {
                LayerType precedent = networkLayers.get(i-1).getLayerType();
                if (!order.antecedentsContain(precedent)) {
                    StringBuilder sb = new StringBuilder();
                    sb.append("\r\nLayer of type ");
                    sb.append("\"" + precedent.toString() + "\" cannot precede layer of type ");
                    sb.append("\"" + type.toString() + "\".");
                    throw new InvalidLayerOrderException(sb.toString());
                }
            }
        }
    }

    /**
     * This is a part of the network validity check. This double checks that there are no negative values for layer output data sizes.
     */
    private void checkDataSizes() {
        for (int i=0; i<networkLayers.size(); i++) {
            int[] outputSize = networkLayers.get(i).getOutputSize();

            for (int os : outputSize) {
                if (os <= 0) {
                    throw new InvalidOutputSizeException("\r\nOutput of layer " + i + " invalid. ");
                }
            }
        }
    }

    /**
     * Sets the training termination conditions. 1 of these must be set before training.
     * @param epoch The number of epochs you wish to train for. -1 to not use this parameter.
     * @param error The minimum error you want to train to. -1 to not use this parameter.
     * @param duration The duration (ms) you want to train for. -1 to not use this parameter.
     */
    public void setTerminationConditions(int epoch, int error, long duration) {
        terminationEpoch = epoch;
        terminationError = error;
        terminationDuration = duration;
    }

    /**
     * Sets the training data.
     * The .csv file must be named "trainingImages.csv" although this will be changeable in the future.
     * @param trainingDir The directory containing the training data.
     */
    public void setTrainingData(String trainingDir) {
        String filePathName = trainingDir + "trainingImages.csv";
        trainingExampleCount = 0;

        try {
            BufferedReader br = new BufferedReader(new FileReader(filePathName));
            String line = null;
            while ((line = br.readLine()) != null) {
                String[] row = line.split(",");
                if (row.length == 2) {
                    trainingCsvData.add(row);
                    trainingExampleCount++;
                }
            }
        } catch (IOException e) {
            System.out.println(Arrays.toString(e.getStackTrace()));
        }
    }

    // Training happens here

    /**
     * This method orchestrates the training of the network. It receives a directory containing the training images.
     * @param trainingDir The directory containing the training images.
     */
    public void startTraining(String trainingDir) {
        if (!networkValid)
            throw new InvalidNetworkException("\r\nEnsure networkValidityCheck() is called before starting training. ");

        initErrorLogging();

        // Set up
        long trainingStartTime = System.currentTimeMillis();
        boolean shouldTerminate = false;
        int currentEpoch = -1;
        double currentError;
	    int firstStart = 0;

	    // Checks
        checkTerminationConditionExists();

        // Training loop
        System.out.println("\r\nStarting training...\r\n");
        while (!shouldTerminate) { // While should train
            currentEpoch++;
            System.out.println("Epoch: " + currentEpoch);

            // For each training example
            for (int i=firstStart%trainingExampleCount; i<trainingExampleCount; i++) {
		        firstStart = 0;
                System.out.println("Example: " + i);

                // Retrieve training data
                String[] rowData = trainingCsvData.get(i);
                String imgSuffix = rowData[0].substring(rowData[0].length()-1, rowData[0].length());

                // Avoiding image slices ending in "-1" as they were not deemed to be useful.
                if (!"1".equals(imgSuffix)) {
                    // Getting the image and class
                    List<DoubleMatrix> inputData = retrieveImageData(trainingDir, rowData[0]);
                    int label = convertLabelToInt(rowData[1]);

                    // Check we have the data
                    if (inputData != null && label > -1) {
                        // Set input layer input data
                        ((LayerInput) networkLayers.get(0)).setInput(inputData);

                        // Forward pass through entire network
                        // Start at 1 because input layer handled differently above
                        for (int j=1; j<networkLayers.size(); j++) {
                            networkLayers.get(j).setInputData(networkLayers.get(j-1).getNeurons());
                        }

                        // Get current total MSE
                        currentError = ((LayerOutput) networkLayers.get(networkLayers.size()-1)).getError(label);
                        logError(currentError, currentEpoch, i);

                        // Backwards pass through entire network
                        List<Neuron> lowerNeurons = networkLayers.get(networkLayers.size()-2).getNeurons();
                        List<Neuron> upperNeurons = null;

                        networkLayers.get(networkLayers.size()-1).calculateDeltas(lowerNeurons, upperNeurons);

                        // Don't need to do anything to input layer
                        // Loop starts at penultimate layer as output layer is handled differently
                        for (int j=networkLayers.size()-2; j>0; j--) {
                            lowerNeurons = networkLayers.get(j-1).getNeurons();
                            upperNeurons = networkLayers.get(j+1).getNeurons();
                            networkLayers.get(j).calculateDeltas(lowerNeurons, upperNeurons);
                        }

                        // Apply deltas
                        for (int j=networkLayers.size()-1; j>0; j--) {
                            networkLayers.get(j).applyDeltas();
                        }

                        // Print stuff
                        if ((i+1)%10 == 0) {
                            System.out.println("Example " + i + ": " + String.format("%.17f", currentError));

                            DoubleMatrix outputs = ((LayerOutput) networkLayers.get(networkLayers.size()-1)).getOutputs();
                            String predicted = extractPrediction(outputs, true);

                            System.out.println("Actual: " + rowData[1] + ", Predicted: " + predicted);
                            System.out.println();
                        }

                        // Termination check
                        shouldTerminate = checkShouldTerminate(trainingStartTime, currentEpoch, currentError);
                        if (shouldTerminate) {
                            errorLogger.flush();
                            saveNetwork();
                            break;
                        }
                    } else {
                        System.out.println("Training example invalid. Skipping.");
                    }
                } else {
                    System.out.println("Skipping image type 1.");
                }
            }
        }

        errorLogger.close();
    }

    /**
     * Converts a .png image to a list of {@link DoubleMatrix}'s containing the pixel values.
     * The method for retrieving the image data is taken from 2 answers on stackoverflow:
     *      - https://stackoverflow.com/a/9470843/3259361
     *      - https://stackoverflow.com/a/10089030/3259361
     *
     * @param trainingDir The directory containing the images.
     * @param imgName The name of the image (without extension).
     * @return A list containing {@link DoubleMatrix}'s containing the pixel values.
     */
    private List<DoubleMatrix> retrieveImageData(String trainingDir, String imgName) {
        List<DoubleMatrix> out = null;

        // https://stackoverflow.com/a/9470843/3259361
        // https://stackoverflow.com/a/10089030/3259361
        int[] pixel;
        try {
            out = new ArrayList<>();

            BufferedImage img = ImageIO.read(new File(trainingDir + imgName + ".png"));
            int width = img.getWidth();
            int height = img.getHeight();

            DoubleMatrix imgData = new DoubleMatrix(height, width);

            for (int y=0; y<height; y++) {
                for (int x=0; x<width; x++) {
                    pixel = img.getRaster().getPixel(x, y, new int[4]);
                    imgData.put(y,x, pixel[0]);
                }
            }

            out.add(imgData);
        } catch (IOException e) {
            System.out.println("Could not read image " + imgName);
        }

        return out;
    }

    /**
     * Takes the String version of a class and converts it to an integer representation.
     * This will change in the future to be more customisable.
     * @param labelStr The String version of the class.
     * @return An int representation of the class.
     */
    private int convertLabelToInt(String labelStr) {
        int label = -1;

        switch (labelStr) {
            case "NL": {
                label = 0;
                break;
            }
            case "MCI": {
                label = 1;
                break;
            }
            case "AD": {
                label = 2;
                break;
            }
        }

        return label;
    }

    /**
     * Uses the outputs from the network to create a prediction.
     * @param outputs The outputs from the network.
     * @param printOutputs A flag whether to print the outputs to the console. In the future this will change to use a logging facade.
     * @return A String version of the predicted class if found, -1 otherwise.
     */
    private String extractPrediction(DoubleMatrix outputs, boolean printOutputs) {
        int predicted = 0;
        double predictedVal = Double.MIN_VALUE;

        for (int o=0; o<outputs.length; o++) {
            if (printOutputs)
                System.out.println(o + ": " + String.format("%.17f", outputs.get(o)));

            if (outputs.get(o) > predictedVal) {
                predictedVal = outputs.get(o);
                predicted = o;
            }
        }

        switch (predicted) {
            case 0: {
                return "NL";
            }
            case 1: {
                return "MCI";
            }
            case 2: {
                return "AD";
            }
        }

        return "-1";
    }

    /**
     * Sets the testing data.
     * The .csv file must be named "testImages.csv" although this will be changeable in the future.
     * @param testingDir The directory containing the training data.
     */
    public void setTestingData(String testingDir) {
        String filePathName = testingDir + "testImages.csv";
        testingExampleCount = 0;

        try {
            BufferedReader br = new BufferedReader(new FileReader(filePathName));
            String line = null;
            while ((line = br.readLine()) != null) {
                String[] row = line.split(",");
                if (row.length == 2) {
                    testingCsvData.add(row);
                    testingExampleCount++;
                }
            }
        } catch (IOException e) {
            System.out.println(Arrays.toString(e.getStackTrace()));
        }
    }

    /**
     * This method orchestrates the testing of the network. It receives a directory containing the testing images.
     * @param testingDir The directory containing the testing images.
     */
    public void startTesting(String testingDir) {
        // Set up
        double currentError;
        int correct = 0;
        int incorrect = 0;
        String[] predictions = new String[testingExampleCount];
        double[] errors = new double[testingExampleCount];

        if (!networkValid)
            throw new InvalidNetworkException("\r\nEnsure networkValidityCheck() is called before starting testing. ");
       
            System.out.println("\r\nStarting testing...\r\n");

        // Testing loop
        for (int i=0; i<testingExampleCount; i++) {
            // Get the data
            String[] rowData = testingCsvData.get(i);
            String imgSuffix = rowData[0].substring(rowData[0].length()-1, rowData[0].length());
            String dgn = rowData[1];

            // Skipping "1" images because they are not useful
            if (!"1".equals(imgSuffix)) {
                List<DoubleMatrix> inputData = retrieveImageData(testingDir, rowData[0]);
                int label = convertLabelToInt(rowData[1]);

                // Check we have the data
                if (inputData != null && label > -1) {
                    System.out.println("Example " + i);

                    // Set the input and forward propagate through the network
                    ((LayerInput) networkLayers.get(0)).setInput(inputData);
                    for (int j=1; j<networkLayers.size(); j++) {
                        networkLayers.get(j).setInputData(networkLayers.get(j-1).getNeurons());
                    }
                    // Get the error of the network
                    currentError = ((LayerOutput) networkLayers.get(networkLayers.size()-1)).getError(label);

                    // Get the outputs and convert to a prediction
                    DoubleMatrix outputs = ((LayerOutput) networkLayers.get(networkLayers.size()-1)).getOutputs();
                    String predicted = extractPrediction(outputs, false);

                    // Store the error and prediction
                    predictions[i] = predicted;
                    errors[i] = currentError;

                    System.out.println("Predicted: " + predicted + ", Actual: " + dgn);

                    if (predicted.equals(dgn)) {
                        correct++;
                    } else {
                        incorrect++;
                    }
                } else {
                    System.out.println("Testing example invalid. Skipping.");
                }
            } else {
                System.out.println("Skipping image type 1.");
            }
        }

        System.out.println("\r\nCorrect: " + correct + ", Incorrect: " + incorrect);
        double perc = ((double) correct / (double) (correct + incorrect)) * 100;
        System.out.println("Correct: " + perc + "%");

        // Save the results of testing
        saveTestResults(predictions, errors);
    }

    /**
     * Save the results from testing to a file in the "./performance-logging/" directory. The directory will be customisable in the future.
     * @param predicted The predictions that the model made.
     * @param errors The errors that the model produced.
     */
    private void saveTestResults(String[] predicted, double[] errors) {
        try {
            String fileName = "NetworkTest-" + networkReference + ".csv";
            String filePath = "./../performance-logging/";

            FileWriter fileWriter = new FileWriter(filePath + fileName, true);
            BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);
            PrintWriter pw = new PrintWriter(bufferedWriter);

            pw.println("Actual;Predicted;Error");

            for (int i=0; i<testingExampleCount; i++) {
                if (predicted[i] != null) {
                    pw.println(testingCsvData.get(i)[1] + ";" + predicted[i] + ";" + errors[i]);
                }
            }

            pw.close();

        } catch (IOException e) {
            System.out.println(Arrays.toString(e.getStackTrace()));
        }
    }

    /**
     * Sets the prediction data.
     * The .csv file must be named "testImages.csv" although this will be changeable in the future.
     * @param predictionDir The directory containing the training data.
     */
    public void setPredictionData(String predictionDir) {
        String filePathName = predictionDir + "testImages.csv";
        predictionExampleCount = 0;

        try {
            BufferedReader br = new BufferedReader(new FileReader(filePathName));
            String line;
            while ((line = br.readLine()) != null) {
                String[] row = line.split(",");
                if (row.length == 2) {
                    predictionCsvData.add(row);
                    predictionExampleCount++;
                }
            }
        } catch (IOException e) {
            System.out.println(e.getStackTrace());
        }
    }

    /**
     * Orchestrates the network to make predictions on images.
     * @param predictionDir The directory containing the images to make predictions on.
     */
    public void startPredicting(String predictionDir) {
        // Set up
        String[] predictions = new String[predictionExampleCount] ;

        if (!networkValid)
            throw new InvalidNetworkException("\r\nEnsure networkValidityCheck() is called before starting testing. ");
       
        System.out.println("\r\nStarting predicting...\r\n");

        // The main loop
        for (int i=0; i<predictionExampleCount; i++) {
            String[] rowData = predictionCsvData.get(i);
            String imgSuffix = rowData[0].substring(rowData[0].length()-1, rowData[0].length());

            // Skipping "1" images because they are not useful.
            if (!"1".equals(imgSuffix)) {
                // Get the image data
                List<DoubleMatrix> inputData = retrieveImageData(predictionDir, rowData[0]);

                // Forward propagation through the network
                ((LayerInput) networkLayers.get(0)).setInput(inputData);
                for (int j=1; j<networkLayers.size(); j++) {
                    networkLayers.get(j).setInputData(networkLayers.get(j-1).getNeurons());
                }

                // Get the output and convert to a prediction and store the prediction.
                DoubleMatrix outputs = ((LayerOutput) networkLayers.get(networkLayers.size()-1)).getOutputs();
                String predicted = extractPrediction(outputs, false);
                System.out.println("Prediction: " + predicted);
                predictions[i] = predicted;

                if (i % 50 == 0) {
                    System.out.println("Example " + i);
                }
            } else {
                System.out.println("Skipping image type 1.");
            }
        }
    }

    /**
     * Checks that the user set a termination condition so that the network doesn't get stuck in an endless training loop.
     */
    private void checkTerminationConditionExists() {
        if (terminationEpoch <= 0 && terminationError <= 0 && terminationDuration <= 0)
            throw new MissingTerminationConditionException("");
    }

    /**
     * Checks whether the network should terminate the training.
     * @param startTime The time that the network started training.
     * @param currentEpoch The current training epoch.
     * @param currentError The current error of the network.
     * @return True if the network should terminate training, false otherwise.
     */
    private boolean checkShouldTerminate(long startTime, int currentEpoch, double currentError) {
        if (terminationEpoch > 0)
            if (currentEpoch >= terminationEpoch)
                return true;

        if (terminationDuration > 0)
            if (System.currentTimeMillis() - startTime > terminationDuration)
                return true;

        if (terminationError >= 0)
            return currentError <= terminationError;

        return false;
    }

    /**
     * Sets up the file for and writes the first line to the error logging file.
     */
    private void initErrorLogging() {
        try {
            String fileName = "NetworkMSE-" + networkReference + ".csv";
            String filePath = "./../error-logging/";

            FileWriter fileWriter = new FileWriter(filePath + fileName, true);
            BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);
            errorLogger = new PrintWriter(bufferedWriter);

            errorLogger.println("Epoch;Example;MSE");
            errorLogger.flush();
        } catch (IOException e) {
            System.out.println("There was a problem creating/accessing the error log file. Error will not be logged.");
        }
    }

    /**
     * Adds a line to the error log.
     * @param mse The network error (MSE).
     * @param epoch The current training epoch.
     * @param example The training example number.
     */
    private void logError(double mse, int epoch, int example) {
        String message = epoch+";"+example+";"+String.format("%.17f", mse);
        errorLogger.println(message);

        // Ensure errors are written to file at least every 50 examples
        if ((example+1) % 50 == 0) {
            errorLogger.flush();
        }
    }

    /**
     * Starts the process of saving a network model.
     */
    private void saveNetwork() {
        try {
            System.out.println("Saving network...");
            NetworkSaver.saveNetwork(this, networkReference); 
        } catch (Exception e) {
            System.out.println("Error saving network model.");
            System.exit(1);
        }
    }
}
