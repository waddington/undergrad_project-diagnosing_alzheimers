package waddington.kai.tests;

import org.junit.runner.RunWith;
import org.junit.runners.Suite;

import waddington.kai.tests.exceptions.*;

@RunWith(Suite.class)
@Suite.SuiteClasses({
    ManagerTestSuite.class,
    HelperTestSuite.class,
    TestExceptionTriggers.class
})

public class TestSuite {}