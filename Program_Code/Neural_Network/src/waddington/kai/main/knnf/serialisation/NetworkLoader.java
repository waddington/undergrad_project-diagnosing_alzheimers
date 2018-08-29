package waddington.kai.main.knnf.serialisation;

import net.lingala.zip4j.core.ZipFile;
import org.apache.commons.io.FileUtils;
import org.jblas.DoubleMatrix;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;
import waddington.kai.main.knnf.LayerActivation;
import waddington.kai.main.knnf.LayerType;
import waddington.kai.main.knnf.NetworkManager;
import waddington.kai.main.knnf.exceptions.MissingLayerTypeSerialisationMethodException;
import waddington.kai.main.knnf.exceptions.MissingNeuronTypeSerialisationMethodException;
import waddington.kai.main.knnf.layers.*;
import waddington.kai.main.knnf.layers.neurons.*;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * This class handles loading a previously created neural network model that was created by this package. The model file should be placed in the appropriate location and the model name should be provided.
 * The model file is a .zip file containing a directory for the model. Inside this directory is a .json file containing information about the {@link NetworkManager}, as well as this, there is a directory for each layer. Inside the layer directories is a .json file containing information for that layer, as well as this there is a directory for the neurons in that layer. Inside this directory there is a .json file for each neuron.
 */
@SuppressWarnings("unchecked")
public class NetworkLoader {

    /**
     * This method is called to load a model file.
     * @param filename The name of the model file, without extension.
     * @return A {@link NetworkManager} instance of the model.
     */
    public static NetworkManager loadNetwork(String filename) {
        /**
         * Create a new instance of {@link NetworkManager}, this will be the loaded model.
         */
        NetworkManager network = new NetworkManager();

        // Settings
        String pathToRoot = "./../";
        String tempDirName = ".loadTemp";
        String saveDirName = "LoadNetwork";

        // Create temp file area
        try {
            createTempFileStructure(pathToRoot, tempDirName);
        } catch (IOException e) {}

        // Unzip file to temp area
        try {
            unpackNetwork(pathToRoot, tempDirName, saveDirName, filename);
        } catch (Exception e) {}

        // Read NetworkManager file and assign hyperparameters
        try {
            network = loadManager(pathToRoot, tempDirName, filename, network);
        } catch (FileNotFoundException e) {}
        catch (IOException e) {}
        catch (ParseException e) {}

        // Read each layer
        try {
            network = getLayers(pathToRoot, tempDirName, filename, network);
        } catch (FileNotFoundException e) {
            System.out.println(e.getStackTrace());
        } catch (IOException e) {
            System.out.println(e.getStackTrace());
        } catch (ParseException e) {
            System.out.println(e.getStackTrace());
        }

        // Read each neuron
        try {
            network = getNeurons(pathToRoot, tempDirName, filename, network);
        } catch (FileNotFoundException e) {
            System.out.println(e.getStackTrace());
        } catch (IOException e) {
            System.out.println(e.getStackTrace());
        } catch (ParseException e) {
            System.out.println(e.getStackTrace());
        }
        
        // remove temp file area
        try {
            removeTempFiles(pathToRoot, tempDirName);
        } catch (IOException e) {}

        return network;
    }

    /**
     * Creates a temporary directory where the model file will be unpacked to and read from.
     * @param pathToRoot A locally set path to the root of the project.
     * @param tempDirName The name of the temporary directory.
     * @throws IOException If cannot create the directory.
     */
    private static void createTempFileStructure(String pathToRoot, String tempDirName)
    throws IOException {
        File tempDir = new File(pathToRoot + tempDirName);

        if (tempDir.exists()) {
            FileUtils.deleteDirectory(tempDir);
        }

        tempDir.mkdir();
    }

    /**
     * The model files are a .zip file, this method unpacks the .zip to the temporary directory created by {@link #createTempFileStructure(String, String)}.
     * @param pathToRoot A locally set path to the root of the project.
     * @param tempDirName The name of the temporary directory.
     * @param saveDirName The name of the directory where the model file is placed.
     * @param fileName The name of the model file, without extension.
     * @throws Exception When the model file cannot be read or when the model file cannot be unpacked.
     */
    private static void unpackNetwork(String pathToRoot, String tempDirName, String saveDirName, String fileName)
    throws Exception {
        ZipFile zipFile = new ZipFile(pathToRoot + saveDirName + "/" + fileName + ".zip");
        zipFile.extractAll(pathToRoot + tempDirName);
    }

    /**
     * Loads the settings of the {@link NetworkManager} instance. The settings are saved in a "NetworkManager.json" file.
     * @param pRoot A locally set path to the root of the project.
     * @param tempDir The name of the temporary directory.
     * @param fileName The name of the model file, without extension.
     * @param network The instance of {@link NetworkManager}.
     * @return The instance of {@link NetworkManager} with hyper-parameters set.
     * @throws FileNotFoundException If cannot find the file.
     * @throws IOException If cannot read the file.
     * @throws ParseException If the json cannot be parsed.
     */
    private static NetworkManager loadManager(String pRoot, String tempDir, String fileName, NetworkManager network)
    throws FileNotFoundException, IOException, ParseException {
        JSONParser parser = new JSONParser();
        JSONObject obj = (JSONObject) parser.parse(new FileReader(pRoot + tempDir + "/" + fileName + "/NetworkManager.json"));

        double learnRate = (double) obj.get("LearningRate");
        double momentum = (double) obj.get("Momentum");

        network.setLearningRate((float) learnRate);
        network.setMomentum((float) momentum);

        return network;
    }

    /**
     * Loads the layers of the model. Each layer is stored in it's own directory which also contains the neuron files as well.
     * The method used to get the number of layers is from stackoverflow:
     *      - https://stackoverflow.com/a/18300155/3259361
     * @param pRoot A locally set path to the root of the project.
     * @param tempDir The name of the temporary directory.
     * @param fileName The name of the model file, without extension.
     * @param network The instance of {@link NetworkManager}.
     * @return The instance of {@link NetworkManager} with the layers added.
     * @throws FileNotFoundException If a layer file cannot be read.
     * @throws IOException If a layer file cannot be found.
     * @throws ParseException If the json data cannot be parsed.
     */
    private static NetworkManager getLayers(String pRoot, String tempDir, String fileName, NetworkManager network)
    throws FileNotFoundException, IOException, ParseException {
        // https://stackoverflow.com/a/18300155/3259361
        // Counts the directories to know how many layers there are
        int numberOfLayers = (int) Files.find(
            Paths.get(pRoot+tempDir+"/"+fileName), 
            1,  // how deep do we want to descend
            (path, attributes) -> attributes.isDirectory()
        ).count() - 1; // '-1' because root is also counted in

        List<Layer> layers = new ArrayList<>();

        // For each layer, get the json data and parse it and add the new layer to the list
        for (int i=0; i<numberOfLayers; i++) {
            String layerPath = pRoot+tempDir+"/"+fileName+"/"+i+"/Layer.json";
            JSONParser parser = new JSONParser();
            JSONObject obj = (JSONObject) parser.parse(new FileReader(layerPath));
            
            String typeString = (String) obj.get("Type");
            LayerType type = LayerType.valueOf(typeString);

            layers.add(getLayer(type, obj));
        }

        network.setLayers(layers);

        return network;
    }

    /**
     * Takes the json data from a layer file, and calls a method to parse the data and convert to a {@link Layer} instance.
     * @param type The type of layer, so that the correct method can be called.
     * @param obj The json data.
     * @return The loaded {@link Layer} instance.
     */
    private static Layer getLayer(LayerType type, JSONObject obj) {
        switch (type) {
            case input: {
                return getInputLayer(obj);
            }
            case conv: {
                return getConvLayer(obj);
            }
            case minPool: {
                return getPoolLayer(obj);
            }
            case maxPool: {
                return getPoolLayer(obj);
            }
            case flatten: {
                return getFlattenLayer(obj);
            }
            case fc: {
                return getFCLayer(obj);
            }
            case output: {
                return getOutputLayer(obj);
            }
            default: {
                throw new MissingLayerTypeSerialisationMethodException("");
            }
        }
    }

    /**
     * Used to load an {@link LayerType#input} layer.
     * @param obj The json data.
     * @return A {@link LayerInput} instance.
     */
    private static Layer getInputLayer(JSONObject obj) {
        int id = getId(obj);
        LayerType type = getType(obj);
        LayerActivation activation = getActivation(obj);
        int numberOfNeurons = getNumberOfNeurons(obj);
        int[] inputSize = getInputSize(obj);
        int[] outputSize = getOutputSize(obj);

        LayerInput layer = new LayerInput(id);
        layer.setInputSize(inputSize);
        layer.setOutoutSize(outputSize);

        return layer;
    }

    /**
     * Used to load an {@link LayerType#conv} layer.
     * @param obj The json data.
     * @return A {@link LayerConvolution} instance.
     */
    private static Layer getConvLayer(JSONObject obj) {
        int id = getId(obj);
        LayerType type = getType(obj);
        LayerActivation activation = getActivation(obj);
        int numberOfNeurons = getNumberOfNeurons(obj);
        int[] inputSize = getInputSize(obj);
        int[] outputSize = getOutputSize(obj);
        int filterSize = getFilterSize(obj);
        int stride = getStride(obj);
        int numberOfFilters = getNumberOfFilters(obj);

        LayerConvolution layer = new LayerConvolution(id, activation);
        layer.setInputSize(inputSize);
        layer.setOutoutSize(outputSize);
        layer.setFilterSize(filterSize);
        layer.setStride(stride);
        layer.setNumberOfFilters(numberOfFilters);

        return layer;
    }

    /**
     * Used to load an {@link LayerType#maxPool} or {@link LayerType#minPool} layer.
     * @param obj The json data.
     * @return A {@link LayerPool} instance.
     */
    private static Layer getPoolLayer(JSONObject obj) {
        int id = getId(obj);
        LayerType type = getType(obj);
        LayerActivation activation = getActivation(obj);
        int numberOfNeurons = getNumberOfNeurons(obj);
        int[] inputSize = getInputSize(obj);
        int[] outputSize = getOutputSize(obj);
        int poolSize = getPoolSize(obj);
        int stride = getStride(obj);

        LayerPool layer = new LayerPool(id, type);
        layer.setInputSize(inputSize);
        layer.setOutoutSize(outputSize);
        layer.setPoolSize(poolSize);
        layer.setStride(stride);

        return layer;
    }

    /**
     * Used to load an {@link LayerType#flatten} layer.
     * @param obj The json data.
     * @return A {@link LayerFlatten} instance.
     */
    private static Layer getFlattenLayer(JSONObject obj) {
        int id = getId(obj);
        LayerType type = getType(obj);
        LayerActivation activation = getActivation(obj);
        int numberOfNeurons = getNumberOfNeurons(obj);
        int[] inputSize = getInputSize(obj);
        int[] outputSize = getOutputSize(obj);

        LayerFlatten layer = new LayerFlatten(id);
        layer.setInputSize(inputSize);
        layer.setOutoutSize(outputSize);
   
        return layer;
    }

    /**
     * Used to load an {@link LayerType#fc} layer.
     * @param obj The json data.
     * @return A {@link LayerFullyConnected} instance.
     */
    private static Layer getFCLayer(JSONObject obj) {
        int id = getId(obj);
        LayerType type = getType(obj);
        LayerActivation activation = getActivation(obj);
        int numberOfNeurons = getNumberOfNeurons(obj);
        int[] inputSize = getInputSize(obj);
        int[] outputSize = getOutputSize(obj);
        int numWights = getNumberOfWeights(obj);

        LayerFullyConnected layer = new LayerFullyConnected(id, activation);
        layer.setInputSize(inputSize);
        layer.setOutoutSize(outputSize);
        layer.setNumberOfWeights(numWights);
        layer.setNumberOfNeurons(numberOfNeurons);
       
        return layer;
    }

    /**
     * Used to load an {@link LayerType#output} layer.
     * @param obj The json data.
     * @return A {@link LayerOutput} instance.
     */
    private static Layer getOutputLayer(JSONObject obj) {
        int id = getId(obj);
        LayerType type = getType(obj);
        LayerActivation activation = getActivation(obj);
        int numberOfNeurons = getNumberOfNeurons(obj);
        int[] inputSize = getInputSize(obj);
        int[] outputSize = getOutputSize(obj);
        int numWeights = getNumberOfWeights(obj);
        int numOutputs = getNumberOfOutputs(obj);

        LayerOutput layer = new LayerOutput(id);
        layer.setInputSize(inputSize);
        layer.setOutoutSize(outputSize);
        layer.setNumberOfWeights(numWeights);
        layer.setNumberOfOutputs(numOutputs);

        return layer;
    }

    /**
     * Loads the neurons of the model. Each neuron is stored in it's own file inside a neuron directory it's parent layer directory.
     * The method used to get the number of neruons in each layer is from stackoverflow:
     *      - https://stackoverflow.com/a/18300155/3259361
     * @param pRoot A locally set path to the root of the project.
     * @param tempDir The name of the temporary directory.
     * @param fileName The name of the model file, without extension.
     * @param network The instance of {@link NetworkManager}.
     * @return The instance of {@link NetworkManager} with the neurons added.
     * @throws FileNotFoundException If a neuron file cannot be read.
     * @throws IOException If a neuron file cannot be found.
     * @throws ParseException If the json data cannot be parsed.
     */
    private static NetworkManager getNeurons(String pRoot, String tempDir, String fileName, NetworkManager network)
    throws FileNotFoundException, IOException, ParseException {
        // https://stackoverflow.com/a/18300155/3259361
        int numberOfLayers = (int) Files.find(
            Paths.get(pRoot+tempDir+"/"+fileName), 
            1,  // how deep do we want to descend
            (path, attributes) -> attributes.isDirectory()
        ).count() - 1; // '-1' because root is also counted in

        for (int i=0; i<numberOfLayers; i++) {
            Layer layer = network.getLayers().get(i);
            // https://stackoverflow.com/a/18300155/3259361
            int numberOfNeurons = (int) Files.find(
                Paths.get(pRoot+tempDir+"/"+fileName + "/" + i + "/neurons/"), 
                1,  // how deep do we want to descend
                (path, attributes) -> attributes.isRegularFile()
            ).count();

            String neuronRootPath = pRoot + tempDir + "/" + fileName + "/" + i + "/neurons/";

            // For each neuron
            for (int j=0; j<numberOfNeurons; j++) {
                String neuronPath = neuronRootPath + j + ".json";

                JSONParser parser = new JSONParser();
                JSONObject obj = (JSONObject) parser.parse(new FileReader(neuronPath));

                LayerType type = getType(obj);

                Neuron neuron = getSpecificNeuron(layer, type, obj);

                layer.addNeuron(neuron);
            }
        }

        return network;
    }

    /**
     * Takes the json data from a neuron file, and calls a method to parse the data and convert to a {@link Neuron} instance.
     * @param layer The {@link Layer} instance that the neurons should be added to.
     * @param type The type of neuron, so that the correct method can be called.
     * @param obj The json data.
     * @return The loaded {@link Neuron} instance.
     */
    private static Neuron getSpecificNeuron(Layer layer, LayerType type, JSONObject obj) {
        switch (type) {
            case input: {
                return getInputNeuron(obj, layer);
            }
            case conv: {
                return getConvNeuron(obj, layer);
            }
            case minPool: {
                return getPoolNeuron(obj, layer);
            }
            case maxPool: {
                return getPoolNeuron(obj, layer);
            }
            case flatten: {
                return getFlattenNeuron(obj, layer);
            }
            case fc: {
                return getFCNeuron(obj, layer);
            }
            case output: {
                return getOutputNeuron(obj, layer);
            }
            default: {
                throw new MissingNeuronTypeSerialisationMethodException("");
            }
        }
    }

    /**
     * Used to load an {@link LayerType#input} neuron.
     * @param obj The json data.
     * @return A {@link NeuronInput} instance.
     */
    private static Neuron getInputNeuron(JSONObject obj, Layer layer) {
        int id = getId(obj);
        LayerType type = getType(obj);
        LayerActivation activation = getActivation(obj);

        NeuronInput neuron = new NeuronInput(id);
        neuron.initMemory(layer.getInputSize(), layer.getOutputSize());

        return neuron;
    }

    /**
     * Used to load an {@link LayerType#conv} neuron.
     * @param obj The json data.
     * @return A {@link NeuronConvolution} instance.
     */
    private static Neuron getConvNeuron(JSONObject obj, Layer layer) {
        int id = getId(obj);
        LayerType type = getType(obj);
        LayerActivation activation = getActivation(obj);
        int filterSize = getFilterSize(obj);
        int stride = getStride(obj);
        double bias = getBias(obj);
        DoubleMatrix filter = getFilter(obj, filterSize);

        NeuronConvolution neuron = new NeuronConvolution(id, filterSize, stride, activation);
        neuron.initMemory(layer.getInputSize(), layer.getOutputSize());
        neuron.setBias(bias);
        neuron.setFilter(filter);

        return neuron;
    }

    /**
     * Used to load an {@link LayerType#minPool} or {@link LayerType#maxPool} neuron.
     * @param obj The json data.
     * @return A {@link NeuronPool} instance.
     */
    private static Neuron getPoolNeuron(JSONObject obj, Layer layer) {
        int id = getId(obj);
        LayerType type = getType(obj);
        LayerActivation activation = getActivation(obj);
        int poolSize = getPoolSize(obj);
        int stride = getStride(obj);

        NeuronPool neuron = new NeuronPool(id, type);
        neuron.initMemory(layer.getInputSize(), layer.getOutputSize());
        neuron.setPoolSize(poolSize);
        neuron.setStride(stride);

        return neuron;
    }

    /**
     * Used to load an {@link LayerType#flatten} neuron.
     * @param obj The json data.
     * @return A {@link NeuronFlatten} instance.
     */
    private static Neuron getFlattenNeuron(JSONObject obj, Layer layer) {
        int id = getId(obj);
        LayerType type = getType(obj);
        LayerActivation activation = getActivation(obj);

        NeuronFlatten neuron = new NeuronFlatten(id);
        int[] dataSize = new int[] {1, 1, 1};
        neuron.initMemory(dataSize, dataSize);

        return neuron;
    }

    /**
     * Used to load an {@link LayerType#fc} neuron.
     * @param obj The json data.
     * @return A {@link NeuronFullyConnected} instance.
     */
    private static Neuron getFCNeuron(JSONObject obj, Layer layer) {
        int id = getId(obj);
        LayerType type = getType(obj);
        LayerActivation activation = getActivation(obj);
        DoubleMatrix weights = getWeights(obj);
        double bias = getBias(obj);

        NeuronFullyConnected neuron = new NeuronFullyConnected(id, activation);
        int[] dataSize = new int[] {1, 1, 1};
        neuron.initMemory(dataSize, dataSize);
        neuron.setBias(bias);
        neuron.setWeights(weights);

        return neuron;
    }

    /**
     * Used to load an {@link LayerType#output} neuron.
     * @param obj The json data.
     * @return A {@link NeuronOutput} instance.
     */
    private static Neuron getOutputNeuron(JSONObject obj, Layer layer) {
        int id = getId(obj);
        LayerType type = getType(obj);
        LayerActivation activation = getActivation(obj);
        DoubleMatrix weights = getWeights(obj);
        double bias = getBias(obj);

        NeuronOutput neuron = new NeuronOutput(id, activation);
        int[] dataSize = new int[] {1, 1, 1};
        neuron.initMemory(dataSize, dataSize);
        neuron.setBias(bias);
        neuron.setWeights(weights);
        
        return neuron;
    }

    /**
     * Used to retrieve the learning rate from the json data.
     * @param obj The json data.
     * @return The learning rate.
     */
    private static float getLearningRate(JSONObject obj) {
        long lr = (Long) obj.get("LearningRate");

        return ((Long) lr).floatValue();
    }

    /**
     * Used to retrieve the momentum from the json data.
     * @param obj The json data.
     * @return The momentum.
     */
    private static float getMomentum(JSONObject obj) {
        long m = (Long) obj.get("Momentum");

        return ((Long) m).floatValue();
    }

    /**
     * Used to retrieve the object ID from the json data.
     * @param obj The json data.
     * @return The object ID.
     */
    private static int getId(JSONObject obj) {
        long id = (Long) obj.get("ID");

        return ((Long) id).intValue();
    }

    /**
     * Used to retrieve the {@link LayerType} from the json data.
     * @param obj The json data.
     * @return The {@link LayerType}.
     */
    private static LayerType getType(JSONObject obj) {
        String typeString = (String) obj.get("Type");

        if ("null".equals(typeString))
            return null;

        return LayerType.valueOf(typeString);
    }

    /**
     * Used to retrive the {@link LayerActivation} from the json data.
     * @param obj The json data.
     * @return The {@link LayerActivation}.
     */
    private static LayerActivation getActivation(JSONObject obj) {
        String activationString = (String) obj.get("Activation");

        if ("null".equals(activationString))
            return null;

        return LayerActivation.valueOf(activationString);
    }

    /**
     * Used to retrieve the number of neurons that a layer has from the json data.
     * @param obj The json data.
     * @return The number of neurons in the layer.
     */
    private static int getNumberOfNeurons(JSONObject obj) {
        long n = (Long) obj.get("NumberOfNeurons");

        return ((Long) n).intValue();
    }

    /**
     * Used to get the size of the input to the layer/neuron from the json data.
     * @param obj The json data.
     * @return An int[] containing the input dimensions in order {Z,Y,X}.
     */
    private static int[] getInputSize(JSONObject obj) {
        int[] out = new int[3];

        JSONArray size = (JSONArray) obj.get("InputSize");
        Iterator<Long> iter = size.iterator();
        int index = 0;
        while (iter.hasNext()) {
            out[index] = iter.next().intValue();
            index++;
        }

        return out;
    }

    /**
     * Used to get the size of the output from the layer/neuron from the json data.
     * @param obj The json data.
     * @return An int[] containing the output dimensions in order {Z,Y,X}.
     */
    private static int[] getOutputSize(JSONObject obj) {
        int[] out = new int[3];

        JSONArray size = (JSONArray) obj.get("OutputSize");
        Iterator<Long> iter = size.iterator();
        int index = 0;
        while (iter.hasNext()) {
            out[index] = iter.next().intValue();
            index++;
        }

        return out;
    }

    /**
     * Used to retrieve the filter (receptive field) size from json data.
     * @param obj The json data.
     * @return The filter size.
     */
    private static int getFilterSize(JSONObject obj) {
        long f = (Long) obj.get("FilterSize");

        return ((Long) f).intValue();
    }

    /**
     * Used to retrieve the stride value from the json data.
     * @param obj The json data.
     * @return The stride value.
     */
    private static int getStride(JSONObject obj) {
        long s = (Long) obj.get("Stride");

        return ((Long) s).intValue();
    }

    /**
     * Used to retrieve the number of filters in a convolution layer from the json data.
     * @param obj The json data.
     * @return The number of filters in a convolution layer.
     */
    private static int getNumberOfFilters(JSONObject obj) {
        long n = (Long) obj.get("NumberOfFilters");

        return ((Long) n).intValue();
    }

    /**
     * Used to retrieve the pool size from the json data.
     * @param obj The json data.
     * @return The pool size.
     */
    private static int getPoolSize(JSONObject obj) {
        long p = (Long) obj.get("PoolSize");

        return ((Long) p).intValue();
    }

    /**
     * Used to retrieve the number of weights a neuron has from the json data.
     * @param obj The json data.
     * @return The number of weights.
     */
    private static int getNumberOfWeights(JSONObject obj) {
        long n = (Long) obj.get("NumberOfWeights");

        return ((Long) n).intValue();
    }

    /**
     * Used to retreive the number of outputs that the output layer has.
     * @param obj The json data.
     * @return The number of outputs.
     */
    private static int getNumberOfOutputs(JSONObject obj) {
        long n = (Long) obj.get("NumberOfOutputs");

        return ((Long) n).intValue();
    }

    /**
     * Used to retrieve the bias value from the json data.
     * @param obj The json data.
     * @return The bias value.
     */
    private static double getBias(JSONObject obj) {
        double b = (Double) obj.get("Bias");

        return b;
    }

    /**
     * Used to get the filter values from the json data.
     * @param obj The json data.
     * @param filterSize The size of the filter so that the {@link DoubleMatrix} that it is stored in can be created to the correct size.
     * @return A {@link DoubleMatrix} containing the filter values.
     */
    private static DoubleMatrix getFilter(JSONObject obj, int filterSize) {
        List<Double> temp = new ArrayList<>();

        JSONArray size = (JSONArray) obj.get("Filter");
        Iterator<String> iter = size.iterator();
        while (iter.hasNext()) {
            temp.add(Double.valueOf(iter.next()));
        }

        DoubleMatrix out = new DoubleMatrix(filterSize, filterSize);
        for (int y=0; y<filterSize; y++) {
            for (int x=0; x<filterSize; x++) {
                out.put(y, x, temp.get((y*filterSize)+x));
            }
        }

        return out;
    }

    /**
     * Used to retrieve the weights from the json data.
     * @param obj The json data.
     * @return A {@link DoubleMatrix} containing the weight values.
     */
    private static DoubleMatrix getWeights(JSONObject obj) {
        List<Double> temp = new ArrayList<>();

        JSONArray size = (JSONArray) obj.get("Weights");
        Iterator<String> iter = size.iterator();
        while (iter.hasNext()) {
            temp.add(Double.valueOf(iter.next()));
        }

        DoubleMatrix out = new DoubleMatrix(temp.size());
        for (int i=0; i<temp.size(); i++) {
            out.put(i, temp.get(i));
        }

        return out;
    }

    /**
     * Removes the temporary directory/files once the model has been loaded.
     * @param pathToRoot A locally set path to the root of the project.
     * @param tempDirName The name of the temporary directory.
     * @throws IOException If the files cannot be deleted.
     */
    private static void removeTempFiles(String pathToRoot, String tempDirName)
    throws IOException {
        FileUtils.deleteDirectory(new File(pathToRoot + tempDirName));
    }
}
