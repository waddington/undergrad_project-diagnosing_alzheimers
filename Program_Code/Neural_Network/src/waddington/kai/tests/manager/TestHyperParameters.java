package waddington.kai.tests.manager;

import waddington.kai.main.knnf.NetworkManager;

import org.junit.*;
import static org.junit.Assert.*;
import static org.hamcrest.CoreMatchers.*;

public class TestHyperParameters {
    private NetworkManager manager;

    @Before
    public void setupOnce() {
        manager = new NetworkManager();
    }

    @Test
    public void testLearningRate() {
        float toSet = 0.005f;
        manager.setLearningRate(toSet);        
        assertEquals("Failure - network learning rate not set correctly.", toSet, NetworkManager.LearningRate, 0.0);
    }

    @Test
    public void testMomentum() {
        float toSet = 0.005f;
        manager.setMomentum(toSet);
        assertEquals("Failure - network momentum not set correctly.", toSet, NetworkManager.Momentum, 0.0);
    }

    @Test
    public void testMaxInitialWeights() {
        float toSet = 0.005f;
        manager.setMaximumInitialWeights(toSet);
        assertEquals("Failure - network maximum initial weights not set correctly.", toSet, NetworkManager.MaximumInitialWeights, 0.0);
    }

    @After
    public void end() {
        manager = null;
    }
}