#!/bin/bash

cd out

java -cp "./../libraries/*:." -Xmx10G org.junit.runner.JUnitCore waddington.kai.tests.TestSuite 

cd ..