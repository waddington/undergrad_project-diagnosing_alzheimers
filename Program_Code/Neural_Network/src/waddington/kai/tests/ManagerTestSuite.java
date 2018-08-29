package waddington.kai.tests;

import org.junit.*;
import org.junit.runner.RunWith;
import org.junit.runners.Suite;

import waddington.kai.tests.manager.*;

@RunWith(Suite.class)
@Suite.SuiteClasses({
    TestHyperParameters.class,
})

public class ManagerTestSuite {

    @BeforeClass
    public static void setup() {
        System.out.println("Testing network manager...");
    }
}