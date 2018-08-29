package waddington.kai.tests.helper;

import org.junit.*;
import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.List;

import static org.hamcrest.CoreMatchers.*;
import org.jblas.*;

import waddington.kai.main.knnf.LayerActivation;
import waddington.kai.main.knnf.NetworkHelper;

public class TestMathsFunctions {

    @Test
    public void testConvOutputSizeCalculator() {
        int numFilters = 5;
        int filterSize = 9;
        int stride = 1;
        int[] inputSize = new int[] {10, 11, 11};

        int[] expected = new int[] {5, 3, 3};
        int[] calculated = NetworkHelper.calculateConvolutionOutputSize(numFilters, filterSize, stride, inputSize);

        assertArrayEquals("Failure - calculated convolution output size incorrect.", expected, calculated);
    }

    @Test
    public void testConvOutputSizeCalculator2() {
        int numFilters = 5;
        int filterSize = 1;
        int stride = 1;
        int[] inputSize = new int[] {10, 11, 11};

        int[] expected = new int[] {5, 11, 11};
        int[] calculated = NetworkHelper.calculateConvolutionOutputSize(numFilters, filterSize, stride, inputSize);

        assertArrayEquals("Failure - calculated convolution output size incorrect with 1x1 filter.", expected, calculated);
    }

    @Test
    public void testPoolOutputSizeCalculator() {
        int poolSize = 2;
        int stride = 2;
        int[] inputSize = new int[] {10, 10, 10};

        int[] expected = new int[] {10, 5, 5};
        int[] calculated = NetworkHelper.calculatePoolOutputSize(poolSize, stride, inputSize);

        assertArrayEquals("Failure - calculated pool output size incorrect.", expected, calculated);
    }

    @Test
    public void testFlattenOutputSizeCalculator() {
        int[] inputSize = new int[] {10, 10, 10};

        int[] expected = new int[] {1, 1, 1000};
        int[] calculated = NetworkHelper.caculateFlattenLayerOutputSize(inputSize);

        assertArrayEquals("Failure - calculated flatten output size incorrect.", expected, calculated);
    }

    @Test
    public void testSumMatrixA() {
        DoubleMatrix m = new DoubleMatrix(3,3, 1,2,3,
                                               2,3,4,
                                               3,4,5);

        double expected = 27;
        double summed = NetworkHelper.sumMatrix(m);

        assertEquals("Failure - summing of matrix incorrect.", expected, summed, 0.0);
    }

    @Test
    public void testSumMatrixB() {
        DoubleMatrix m = new DoubleMatrix(3,3, 1,2,3,
                                               2,-8,4,
                                               3,4,5);

        double expected = 16;
        double summed = NetworkHelper.sumMatrix(m);

        assertEquals("Failure - summing of matrix incorrect.", expected, summed, 0.0);
    }

    @Test
    public void testSumMatrices() {
        DoubleMatrix a = new DoubleMatrix(3,3, 1,2,3,
                                               2,3,4,
                                               3,4,5);
        DoubleMatrix b = new DoubleMatrix(3,3, 1,2,3,
                                               2,-8,4,
                                               3,4,5);
        List<DoubleMatrix> list = new ArrayList<>();
        list.add(a);
        list.add(b);

        DoubleMatrix returned = NetworkHelper.sumMatrices(list);
        double summed = NetworkHelper.sumMatrix(returned);
        double expected = 43;

        assertEquals("Failure - adding list of matrices incorrect.", expected, summed, 0.0);
    }

    @Test
    public void testSubMatrixCreationA() {
        DoubleMatrix a = new DoubleMatrix(3,3, 1,2,3,
                                               2,3,4,
                                               3,4,5);
        DoubleMatrix sub = NetworkHelper.createSubMatrix(a, 1, 0, 2, 2, 1);
        DoubleMatrix expected = new DoubleMatrix(2,2, 2,3,
                                                      3,4);
        
        assertEquals("Failure - creating sub matrix does not work correctly.", expected, sub);
    }

    @Test
    public void testMatrixActivationA() {
        DoubleMatrix m = new DoubleMatrix(3,3, 1,2,3,
                                               2,-8,4,
                                               3,4,5);
        LayerActivation activation = LayerActivation.relu;
        DoubleMatrix returned = NetworkHelper.applyActivation(activation, m);

        DoubleMatrix expected = new DoubleMatrix(3,3, 1,2,3,
                                                      2,0,4,
                                                      3,4,5);

        assertEquals("Failure - applying relu activation to matrix does not work correctly.", expected, returned);
    }

    @Test
    public void testMatrixActivationB() {
        DoubleMatrix m = new DoubleMatrix(3,3, 1,2,3,
                                               2,-8,4,
                                               3,4,5);
        LayerActivation activation = LayerActivation.lrelu;
        DoubleMatrix returned = NetworkHelper.applyActivation(activation, m);

        DoubleMatrix expected = new DoubleMatrix(3,3, 1,2,3,
                                                      2,-0.08,4,
                                                      3,4,5);

        assertEquals("Failure - applying lrelu activation to matrix does not work correctly.", expected, returned);
    }

    @Test
    public void testSigmoid() {
        double a = 1.23;
        double returned = NetworkHelper.sigmoid(a);
        double expected = 0.7738186;

        assertEquals("Failure - sigmoid function incorrect.", expected, returned, 0.000001);
    }

    @Test
    public void testSigmoid2() {
        double a = -1.23;
        double returned = NetworkHelper.sigmoid(a);
        double expected = 0.226181;

        assertEquals("Failure - sigmoid function incorrect for negative numbers.", expected, returned, 0.000001);
    }

    @Test
    public void testTanh() {
        double a = 1.23;
        double returned = NetworkHelper.tanh(a);
        double expected = 0.842579;

        assertEquals("Failure - tanh function incorrect.", expected, returned, 0.000001);
    }

    @Test
    public void testTanh2() {
        double a = -1.23;
        double returned = NetworkHelper.tanh(a);
        double expected = -0.842579;

        assertEquals("Failure - tanh function incorrect for negative numbers.", expected, returned, 0.000001);
    }

    @Test
    public void testRelu() {
        double a = 1.23;
        double returned = NetworkHelper.relu(a, 0);
        double expected = 1.23;

        assertEquals("Failure - tanh function incorrect.", expected, returned, 0.000001);
    }

    @Test
    public void testRelu2() {
        double a = -1.23;
        double returned = NetworkHelper.relu(a, 0);
        double expected = 0;

        assertEquals("Failure - tanh function incorrect for negative numbers.", expected, returned, 0.000001);
    }

    @Test
    public void testSigmoidDerivative() {
        double a = 1.23;
        double returned = NetworkHelper.sigmoidDerivative(a);
        double expected = 0.175023;

        assertEquals("Failure - sigmoid derivative function incorrect.", expected, returned, 0.000001);
    }

    @Test
    public void testTanhDerivative() {
        double a = 1.23;
        double returned = NetworkHelper.tanhDerivative(a);
        double expected = 0.290060;

        assertEquals("Failure - tanh derivative function incorrect.", expected, returned, 0.000001);
    }

    @Test
    public void testSigmoidDerivative2() {
        double a = -1.23;
        double returned = NetworkHelper.sigmoidDerivative(a);
        double expected = 0.175023;

        assertEquals("Failure - sigmoid derivative function incorrect.", expected, returned, 0.000001);
    }

    @Test
    public void testTanhDerivative2() {
        double a = -1.23;
        double returned = NetworkHelper.tanhDerivative(a);
        double expected = 0.290060;

        assertEquals("Failure - tanh derivative function incorrect.", expected, returned, 0.000001);
    }

    @Test
    public void testPoolLocationEncoder() {
        int y = 3;
        int x = 5;
        int inRows = 7;
        int inCols = 7;

        int expected = 1351;
        int calculated = NetworkHelper.poolEncodeLocation(y, x, inRows, inCols);

        assertEquals("Failure - pool location encoder incorrect.", expected, calculated);
    }

    @Test
    public void testPoolLocationDecoder() {
        int encoded = 1351;
        int inRows = 7;

        int[] expected = new int[] {3, 5};
        int[] calculated = NetworkHelper.poolDecodeLocation(encoded, inRows);

        assertArrayEquals("Failure - pool location decoder incorrect.", expected, calculated);
    }
}