package waddington.kai.tests;

import org.junit.*;
import org.junit.runner.RunWith;
import org.junit.runners.Suite;

import waddington.kai.tests.helper.*;

@RunWith(Suite.class)
@Suite.SuiteClasses({
    TestMathsFunctions.class
})

public class HelperTestSuite {

    @BeforeClass
    public static void setup() {
        System.out.println("Testing network helper...");
    }
}