#!/bin/bash

cd src

javac -d "./../out/" -cp "./../libraries/*:." "./waddington/kai/main/Main.java"

javac -d "./../out/" -cp "./../libraries/*:." "./waddington/kai/tests/TestSuite.java"

cd ..