#!/bin/bash

cd out

java -cp "./../libraries/*:." -Xmx8G waddington.kai.main.Main

cd ..
