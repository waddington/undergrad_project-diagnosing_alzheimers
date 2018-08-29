package waddington.kai.main.knnf.serialisation;

import net.lingala.zip4j.core.ZipFile;
import net.lingala.zip4j.model.ZipParameters;
import net.lingala.zip4j.util.Zip4jConstants;
import org.apache.commons.io.FileUtils;
import org.jblas.DoubleMatrix;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import waddington.kai.main.knnf.LayerActivation;
import waddington.kai.main.knnf.LayerType;
import waddington.kai.main.knnf.NetworkManager;
import waddington.kai.main.knnf.exceptions.MissingLayerTypeSerialisationMethodException;
import waddington.kai.main.knnf.exceptions.MissingNeuronTypeSerialisationMethodException;
import waddington.kai.main.knnf.layers.*;
import waddington.kai.main.knnf.layers.neurons.*;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

/**
 * This class handles saving a neural network model that was created by this package. The model file will be placed in the appropriate location.
 * The model file will be a .zip file containing a directory for the model. Inside this directory is a .json file containing information about the {@link NetworkManager}, as well as this, there is a directory for each layer. Inside the layer directories is a .json file containing information for that layer, as well as this there is a directory for the neurons in that layer. Inside this directory there is a .json file for each neuron.
 */
@SuppressWarnings("unchecked")
public class NetworkSaver {

    /**
     * This method is called to save a network model.
     * @param networkManager The {@link NetworkManager} instance to save.
     * @param networkReference The name that the model will be saved as. This is currently auto-generated.
     * @throws IOException When the network cannot be saved.
     * @throws Exception When various directories and files cannot be created.
     */
    public static void saveNetwork(NetworkManager networkManager, String networkReference)
    throws IOException, Exception {
        String pathToRoot = "./../";
        String tempDirName = ".saveTemp";
        String saveDirName = "Network-" + networkReference;

        createTempFileStructure(networkManager, pathToRoot, tempDirName, saveDirName);
        serialiseNetwork(networkManager, pathToRoot, tempDirName, saveDirName);

        createZip(pathToRoot, tempDirName, saveDirName);
        removeTempFiles(pathToRoot, tempDirName);
    }

    /**
     * Creates a temporary directory structure where all of the model save files will be located before being zipped up. Creates the main directory as well as all sub-directories for layers and neurons.
     * @param networkManager The {@link NetworkManager} instance to save.
     * @param pathToRoot A locally set path to the root of the project.
     * @param tempDirName The name of the temporary directory.
     * @param saveDirName The directory where the model save file will be placed.
     * @throws IOException When the directories cannot be created.
     */
    private static void createTempFileStructure(NetworkManager networkManager, String pathToRoot, String tempDirName, String saveDirName)
    throws IOException {
        // Create base dirs
        File tempDir = new File(pathToRoot + tempDirName);
        File saveDir = new File(pathToRoot + tempDirName + "/" + saveDirName);

        if (tempDir.exists()) {
            FileUtils.deleteDirectory(tempDir);
        }

        tempDir.mkdir();
        saveDir.mkdir();

        // Create layer and neuron dirs
        int numberOfLayers = networkManager.getLayers().size();
        for (int i=0; i<numberOfLayers; i++) {
            File layerDir = new File(pathToRoot + tempDirName + "/" + saveDirName + "/" + i);
            File neuronDir = new File(pathToRoot + tempDirName + "/" + saveDirName + "/" + i + "/neurons");
            
            layerDir.mkdir();
            neuronDir.mkdir();
        }
    }

    /**
     * Orchestrates writing everything to a file.
     * @param networkManager The model to be saved.
     * @param pathToRoot A locally set path to the root of the project.
     * @param tempDirName The name of the temporary directory.
     * @param saveDirName The directory where the model save file will be placed.
     * @throws IOException When files cannot be written.
     */
    private static void serialiseNetwork(NetworkManager networkManager, String pathToRoot, String tempDirName, String saveDirName)
    throws IOException {
        // NetworkManager
        serialiseNetworkManager(networkManager, pathToRoot, tempDirName, saveDirName);

        // Layers
        serialiseLayers(networkManager, pathToRoot, tempDirName, saveDirName);

        // Neurons
        serialiseNeurons(networkManager, pathToRoot, tempDirName, saveDirName);
    }

    /**
     * Saves the {@link NetworkManager} hyper-paremeters to a json file.
     * @param networkManager The model to be saved.
     * @param pathToRoot A locally set path to the root of the project.
     * @param tempDirName The name of the temporary directory.
     * @param saveDirName The directory where the model save file will be placed.
     * @throws IOException When the file cannot be written.
     */
    private static void serialiseNetworkManager(NetworkManager networkManager, String pathToRoot, String tempDirName, String saveDirName)
    throws IOException {
        JSONObject obj = new JSONObject();
        obj.put("LearningRate", NetworkManager.LearningRate);
        obj.put("Momentum", NetworkManager.Momentum);

        File file = new File(pathToRoot + tempDirName + "/" + saveDirName + "/NetworkManager.json");
        file.createNewFile();
        FileWriter fw = new FileWriter(file);
        BufferedWriter bw = new BufferedWriter(fw);
        bw.write(obj.toJSONString());
        bw.close();
    }

	/**
	 * Saves all of the layers in the model to their corresponding files in their corresponding directories.
	 * @param networkManager The model to be saved.
	 * @param pathToRoot A locally set path to the root of the project.
	 * @param tempDirName The name of the temporary directory.
	 * @param saveDirName The directory where the model save file will be placed.
	 * @throws IOException When the file cannot be written.
	 */
    private static void serialiseLayers(NetworkManager networkManager, String pathToRoot, String tempDirName, String saveDirName)
    throws IOException {
        List<Layer> networkLayers = networkManager.getLayers();

        // For each layer...
        for (int i=0; i<networkLayers.size(); i++) {
        	// Get all the common info (common between all layer types)
            Layer layer = networkLayers.get(i);
            int id = layer.getId();
            String type = layer.getLayerType().name();
            LayerActivation activationA = layer.getLayerActivation();
            String activation = (activationA != null) ? activationA.name() : "null";
            int[] inputSize = layer.getInputSize();
            int[] outputSize = layer.getOutputSize();
            int numberOfNeurons = layer.getNumberOfNeurons();

            // Convert info to json
            JSONObject obj = new JSONObject();
            obj.put("ID", id);
            obj.put("Type", type);
            obj.put("Activation", activation);
            obj.put("NumberOfNeurons", numberOfNeurons);

            JSONArray inSize = new JSONArray();
            for (int in: inputSize) inSize.add(in);

            JSONArray outSize = new JSONArray();
            for (int out: outputSize) outSize.add(out);

            obj.put("InputSize", inSize);
            obj.put("OutputSize", outSize);

            // Get layer-type-specific information as json data
            obj = getLayerSpecifics(layer, obj);

            // Write to file
            File file = new File(pathToRoot + tempDirName + "/" + saveDirName + "/" + i + "/Layer.json");
            file.createNewFile();
            FileWriter fw = new FileWriter(file);
            BufferedWriter bw = new BufferedWriter(fw);
            bw.write(obj.toJSONString());
            bw.close();
        }
    }

	/**
	 * Used to get information that is specific to layer types.
	 * @param layer The layer in question.
	 * @param obj The {@link JSONObject} to save the information to.
	 * @return The {@link JSONObject} with the layer-type-specific information.
	 */
    private static JSONObject getLayerSpecifics(Layer layer, JSONObject obj) {
        LayerType type = layer.getLayerType();

        switch (type) {
            case input: {
                return getInputLayerSpecifics(layer, obj);
            }
            case conv: {
                return getConvLayerSpecifics(layer, obj);
            }
            case minPool: {
                return getPoolLayerSpecifics(layer, obj);
            }
            case maxPool: {
                return getPoolLayerSpecifics(layer, obj);
            }
            case flatten: {
                return getFlattenLayerSpecifics(layer, obj);
            }
            case fc: {
                return getFCLayerSpecifics(layer, obj);
            }
            case output: {
                return getOutputLayerSpecifics(layer, obj);
            }
            default: {
                throw new MissingLayerTypeSerialisationMethodException("");
            }
        }
    }

	/**
	 * Get {@link LayerInput} specific information as json data.
	 * @param layer The layer instance.
	 * @param obj The {@link JSONObject} to add the data to.
	 * @return The {@link JSONObject} with the layer specific data added.
	 */
    private static JSONObject getInputLayerSpecifics(Layer layer, JSONObject obj) {
        return obj;
    }

	/**
	 * Get {@link LayerConvolution} specific information as json data.
	 * @param layer The layer instance.
	 * @param obj The {@link JSONObject} to add the data to.
	 * @return The {@link JSONObject} with the layer specific data added.
	 */
    private static JSONObject getConvLayerSpecifics(Layer layer, JSONObject obj) {
        LayerConvolution layerC = (LayerConvolution) layer;
        int filterSize = layerC.getFilterSize();
        int stride = layerC.getStride();
        int numFilters = layerC.getNumberOfFilters();

        obj.put("FilterSize", filterSize);
        obj.put("Stride", stride);
        obj.put("NumberOfFilters", numFilters);

        return obj;
    }

	/**
	 * Get {@link LayerPool} specific information as json data.
	 * @param layer The layer instance.
	 * @param obj The {@link JSONObject} to add the data to.
	 * @return The {@link JSONObject} with the layer specific data added.
	 */
    private static JSONObject getPoolLayerSpecifics(Layer layer, JSONObject obj) {
        LayerPool layerP = (LayerPool) layer;
        int poolSize = layerP.getPoolSize();
        int stride = layerP.getStride();

        obj.put("PoolSize", poolSize);
        obj.put("Stride", stride);

        return obj;
    }

	/**
	 * Get {@link LayerFlatten} specific information as json data.
	 * @param layer The layer instance.
	 * @param obj The {@link JSONObject} to add the data to.
	 * @return The {@link JSONObject} with the layer specific data added.
	 */
    private static JSONObject getFlattenLayerSpecifics(Layer layer, JSONObject obj) {
        return obj;
    }

	/**
	 * Get {@link LayerFullyConnected} specific information as json data.
	 * @param layer The layer instance.
	 * @param obj The {@link JSONObject} to add the data to.
	 * @return The {@link JSONObject} with the layer specific data added.
	 */
    private static JSONObject getFCLayerSpecifics(Layer layer, JSONObject obj) {
        LayerFullyConnected layerFC = (LayerFullyConnected) layer;
        int numWeights = layerFC.getNumberOfWeights();

        obj.put("NumberOfWeights", numWeights);

        return obj;
    }

	/**
	 * Get {@link LayerOutput} specific information as json data.
	 * @param layer The layer instance.
	 * @param obj The {@link JSONObject} to add the data to.
	 * @return The {@link JSONObject} with the layer specific data added.
	 */
    private static JSONObject getOutputLayerSpecifics(Layer layer, JSONObject obj) {
        LayerOutput layerO = (LayerOutput) layer;
        int numOutputs = layerO.getNumberOfOutputs();
        int numWeights = layerO.getNumberOfWeights();

        obj.put("NumberOfWeights", numWeights);
        obj.put("NumberOfOutputs", numOutputs);

        return obj;
    }

	/**
	 * Saves all of the neurons in the model to their corresponding files in their corresponding directories.
	 * @param networkManager The model to be saved.
	 * @param pathToRoot A locally set path to the root of the project.
	 * @param tempDirName The name of the temporary directory.
	 * @param saveDirName The directory where the model save file will be placed.
	 * @throws IOException When the file cannot be written.
	 */
    private static void serialiseNeurons(NetworkManager networkManager, String pathToRoot, String tempDirName, String saveDirName)
    throws IOException {
        List<Layer> networkLayers = networkManager.getLayers();

        // for each layer...
        for (int i=0; i<networkLayers.size(); i++) {
            Layer layer = networkLayers.get(i);
            List<Neuron> layerNeurons = layer.getNeurons();

            // for each neuron...
            for (int j=0; j<layerNeurons.size(); j++) {
                Neuron neuron = layerNeurons.get(j);

                // Get common information
                int id = neuron.getId();
                String type = neuron.getType().name();
                LayerActivation activationA = neuron.getActivation();
                String activation = (activationA != null) ? activationA.name() : "null";

                // Convert the info to json data
                JSONObject obj = new JSONObject();
                obj.put("ID", id);
                obj.put("Type", type);
                obj.put("Activation", activation);

                // Get neuron type specific information
                obj = getNeuronSpecifics(neuron, obj);

                // Write to file
                File file = new File(pathToRoot + tempDirName + "/" + saveDirName + "/" + i + "/neurons/" + j + ".json");
                file.createNewFile();
                FileWriter fw = new FileWriter(file);
                BufferedWriter bw = new BufferedWriter(fw);
                bw.write(obj.toJSONString());
                bw.close();
            }
        }
    }

	/**
	 * Used to get information that is specific to neuron types.
	 * @param neuron The layer in question.
	 * @param obj The {@link JSONObject} to save the information to.
	 * @return The {@link JSONObject} with the layer-type-specific information.
	 */
    private static JSONObject getNeuronSpecifics(Neuron neuron, JSONObject obj) {
        LayerType type = neuron.getType();

        switch (type) {
            case input: {
                return getInputNeuronSpecifics(neuron, obj);
            }
            case conv: {
                return getConvNeuronSpecifics(neuron, obj);
            }
            case minPool: {
                return getPoolNeuronSpecifics(neuron, obj);
            }
            case maxPool: {
                return getPoolNeuronSpecifics(neuron, obj);
            }
            case flatten: {
                return getFlattenNeuronSpecifics(neuron, obj);
            }
            case fc: {
                return getFCNeuronSpecifics(neuron, obj);
            }
            case output: {
                return getOutputNeuronSpecifics(neuron, obj);
            }
            default: {
                throw new MissingNeuronTypeSerialisationMethodException("");
            }
        }
    }

	/**
	 * Get {@link NeuronInput} specific information as json data.
	 * @param neuron The neuron instance.
	 * @param obj The {@link JSONObject} to add the data to.
	 * @return The {@link JSONObject} with the neuron specific data added.
	 */
    private static JSONObject getInputNeuronSpecifics(Neuron neuron, JSONObject obj) {
        return obj;
    }

	/**
	 * Get {@link NeuronConvolution} specific information as json data.
	 * @param neuron The neuron instance.
	 * @param obj The {@link JSONObject} to add the data to.
	 * @return The {@link JSONObject} with the neuron specific data added.
	 */
    private static JSONObject getConvNeuronSpecifics(Neuron neuron, JSONObject obj) {
        NeuronConvolution neuronC = (NeuronConvolution) neuron;

        int filterSize = neuronC.getFilterSize();
        int stride = neuronC.getStride();
        DoubleMatrix filter = neuronC.getFilter();
        double bias = neuronC.getBias();

        obj.put("FilterSize", filterSize);
        obj.put("Stride", stride);
        obj.put("Bias", bias);

        JSONArray filterArray = new JSONArray();
        for (int i=0; i<filter.length; i++) {
            filterArray.add(String.format("%.20f", filter.get(i)));
        }
        obj.put("Filter", filterArray);

        return obj;
    }

	/**
	 * Get {@link NeuronPool} specific information as json data.
	 * @param neuron The neuron instance.
	 * @param obj The {@link JSONObject} to add the data to.
	 * @return The {@link JSONObject} with the neuron specific data added.
	 */
    private static JSONObject getPoolNeuronSpecifics(Neuron neuron, JSONObject obj) {
        NeuronPool neuronP = (NeuronPool) neuron;

        int poolSize = neuronP.getPoolSize();
        int stride = neuronP.getStride();

        obj.put("PoolSize", poolSize);
        obj.put("Stride", stride);

        return obj;
    }

	/**
	 * Get {@link NeuronFlatten} specific information as json data.
	 * @param neuron The neuron instance.
	 * @param obj The {@link JSONObject} to add the data to.
	 * @return The {@link JSONObject} with the neuron specific data added.
	 */
    private static JSONObject getFlattenNeuronSpecifics(Neuron neuron, JSONObject obj) {
        return obj;
    }

	/**
	 * Get {@link NeuronFullyConnected} specific information as json data.
	 * @param neuron The neuron instance.
	 * @param obj The {@link JSONObject} to add the data to.
	 * @return The {@link JSONObject} with the neuron specific data added.
	 */
    private static JSONObject getFCNeuronSpecifics(Neuron neuron, JSONObject obj) {
        NeuronFullyConnected neuronF = (NeuronFullyConnected) neuron;

        double bias = neuronF.getBias();
        DoubleMatrix weights = neuronF.getWeights();

        JSONArray weightsArray = new JSONArray();
        for (int i=0; i<weights.length; i++) {
            weightsArray.add(String.format("%.20f", weights.get(i)));
        }
        obj.put("Weights", weightsArray);

        obj.put("Bias", bias);

        return obj;
    }

	/**
	 * Get {@link NeuronOutput} specific information as json data.
	 * @param neuron The neuron instance.
	 * @param obj The {@link JSONObject} to add the data to.
	 * @return The {@link JSONObject} with the neuron specific data added.
	 */
    private static JSONObject getOutputNeuronSpecifics(Neuron neuron, JSONObject obj) {
        NeuronOutput neuronO = (NeuronOutput) neuron;

        double bias = neuronO.getBias();
        DoubleMatrix weights = neuronO.getWeights();

        JSONArray weightsArray = new JSONArray();
        for (int i=0; i<weights.length; i++) {
            weightsArray.add(String.format("%.20f", weights.get(i)));
        }
        obj.put("Weights", weightsArray);
        obj.put("Bias", bias);

        return obj;
    }

	/**
	 * Creates a .zip file of the model save directory and moves it to the correct directory.
	 * @param pathToRoot A locally set path to the root of the project.
	 * @param tempDirName The name of the temporary directory.
	 * @param saveDirName The directory where the model save file will be placed.
	 * @throws Exception When the file cannot be zipped or moved.
	 */
    private static void createZip(String pathToRoot, String tempDirName, String saveDirName)
    throws Exception {
        ZipFile zipFile = new ZipFile(pathToRoot + "NetworkSaves/" + saveDirName + ".zip");

        ZipParameters parameters = new ZipParameters();
        parameters.setCompressionMethod(Zip4jConstants.COMP_DEFLATE);
        parameters.setCompressionLevel(Zip4jConstants.DEFLATE_LEVEL_ULTRA);

        zipFile.addFolder(pathToRoot + tempDirName + "/" + saveDirName, parameters);
    }

	/**
	 * Removes the temporary directory/files once the model has been saved.
	 * @param pathToRoot A locally set path to the root of the project.
	 * @param tempDirName The name of the temporary directory.
	 * @throws IOException If the files cannot be deleted.
	 */
    private static void removeTempFiles(String pathToRoot, String tempDirName)
    throws IOException {
        FileUtils.deleteDirectory(new File(pathToRoot + tempDirName));
    }
}