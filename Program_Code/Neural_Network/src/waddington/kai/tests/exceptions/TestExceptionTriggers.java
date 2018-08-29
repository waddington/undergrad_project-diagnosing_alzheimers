package waddington.kai.tests.exceptions;

import org.junit.*;

import waddington.kai.main.knnf.NetworkManager;
import waddington.kai.main.knnf.layers.LayerFactory;
import waddington.kai.main.knnf.exceptions.*;

public class TestExceptionTriggers {
	NetworkManager networkManager;

	@Before
	public void before() {
		networkManager = new NetworkManager();
		LayerFactory.numberOfLayers = 0;

		// Hyperparameters
        networkManager.setLearningRate(0.001f);
        networkManager.setMaximumInitialWeights(0.001f);
		networkManager.setMomentum(0.001f);
	}

	@Test(expected = InvalidLayerOrderException.class)
	public void testLayerOrderA() {
		networkManager.addInput(10, 10, 1);
		networkManager.addConv(5, 5, 1, "lrelu");

		networkManager.networkValidityCheck();
	}

	@Test(expected = InvalidLayerOrderException.class)
	public void testLayerOrderB() {
		networkManager.addInput(10, 10, 1);
		networkManager.addFC(10, "tanh");

		networkManager.networkValidityCheck();
	}

	@Test(expected = InvalidLayerOrderException.class)
	public void testLayerOrderC() {
		networkManager.addInput(10, 10, 1);
		networkManager.addOutput(5);

		networkManager.networkValidityCheck();
	}

	@Test(expected = InvalidLayerOrderException.class)
	public void testLayerOrderD() {
		networkManager.addInput(10, 10, 1);
		networkManager.addFlatten();

		networkManager.networkValidityCheck();
	}

	@Test
	public void testLayerOrderE() {
		networkManager.addInput(10, 10, 1);
		networkManager.addConv(50, 5, 1, "lrelu");
		networkManager.addFlatten();
		networkManager.addOutput(5);

		networkManager.networkValidityCheck();
	}

	@Test(expected = InvalidOutputSizeException.class)
	public void testOutputSizes() {
		networkManager.addInput(10, 10, 1);
		networkManager.addConv(10, 20, 1, "lrelu");
		networkManager.addFlatten();
		networkManager.addOutput(3);

		networkManager.networkValidityCheck();
	}

	@Test
	public void testOutputSizesB() {
		networkManager.addInput(10, 10, 1);
		networkManager.addConv(10, 3, 1, "lrelu");
		networkManager.addFlatten();
		networkManager.addOutput(3);

		networkManager.networkValidityCheck();
	}

	@Test(expected = UnknownActivationTypeException.class)
	public void testActivationMethodsA() {
		networkManager.addInput(10, 10, 1);
		networkManager.addConv(10, 3, 1, "lrelua");
		networkManager.addFlatten();
		networkManager.addOutput(3);

		networkManager.networkValidityCheck();
	}

	@Test(expected = UnknownActivationTypeException.class)
	public void testActivationMethodsB() {
		networkManager.addInput(10, 10, 1);
		networkManager.addConv(10, 3, 1, "lRelu");
		networkManager.addFlatten();
		networkManager.addOutput(3);

		networkManager.networkValidityCheck();
	}

	@Test
	public void testActivationMethodsC() {
		networkManager.addInput(10, 10, 1);
		networkManager.addConv(10, 3, 1, "linear");
		networkManager.addFlatten();
		networkManager.addOutput(3);

		networkManager.networkValidityCheck();
	}

	@Test
	public void testActivationMethodsD() {
		networkManager.addInput(10, 10, 1);
		networkManager.addConv(10, 3, 1, "sigmoid");
		networkManager.addFlatten();
		networkManager.addOutput(3);

		networkManager.networkValidityCheck();
	}

	@Test
	public void testActivationMethodsE() {
		networkManager.addInput(10, 10, 1);
		networkManager.addConv(10, 3, 1, "tanh");
		networkManager.addFlatten();
		networkManager.addOutput(3);

		networkManager.networkValidityCheck();
	}

	@Test
	public void testActivationMethodsF() {
		networkManager.addInput(10, 10, 1);
		networkManager.addConv(10, 3, 1, "relu");
		networkManager.addFlatten();
		networkManager.addOutput(3);

		networkManager.networkValidityCheck();
	}

	@Test
	public void testActivationMethodsG() {
		networkManager.addInput(10, 10, 1);
		networkManager.addConv(10, 3, 1, "lrelu");
		networkManager.addFlatten();
		networkManager.addOutput(3);

		networkManager.networkValidityCheck();
	}

	@Test
	public void testPoolTypesA() {
		networkManager.addInput(10, 10, 1);
		networkManager.addPool(2, 2, "max");
		networkManager.addFlatten();
		networkManager.addOutput(3);

		networkManager.networkValidityCheck();
	}

	@Test
	public void testPoolTypesB() {
		networkManager.addInput(10, 10, 1);
		networkManager.addPool(2, 2, "min");
		networkManager.addFlatten();
		networkManager.addOutput(3);

		networkManager.networkValidityCheck();
	}

	@Test(expected = UnknownPoolTypeException.class)
	public void testPoolTypesC() {
		networkManager.addInput(10, 10, 1);
		networkManager.addPool(2, 2, "avg");
		networkManager.addFlatten();
		networkManager.addOutput(3);

		networkManager.networkValidityCheck();
	}

	@Test(expected = UnknownPoolTypeException.class)
	public void testPoolTypesD() {
		networkManager.addInput(10, 10, 1);
		networkManager.addPool(2, 2, "abc");
		networkManager.addFlatten();
		networkManager.addOutput(3);

		networkManager.networkValidityCheck();
	}

	@Test(expected = MissingTerminationConditionException.class)
	public void testTerminationsA() {
		networkManager.addInput(10, 10, 1);
		networkManager.addPool(2, 2, "max");
		networkManager.addFlatten();
		networkManager.addOutput(3);
		networkManager.networkValidityCheck();

		networkManager.setTrainingData("./../mri-png/");
		networkManager.startTraining("./../mri-png/");
	}

	@Test(expected = IndexOutOfBoundsException.class)
	public void testTerminationsB() {
		networkManager.addInput(10, 10, 1);
		networkManager.addPool(2, 2, "max");
		networkManager.addFlatten();
		networkManager.addOutput(3);
		networkManager.networkValidityCheck();

		networkManager.setTrainingData("./../mri-png/");
		networkManager.setTerminationConditions(1, -1, -1);
		networkManager.startTraining("./../mri-png/");
	}
}