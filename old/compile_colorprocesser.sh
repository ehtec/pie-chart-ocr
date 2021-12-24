#!/bin/bash

g++ -I color/src -c -fPIC colorprocesser.cpp -o colorprocesser.o -O3
g++ -shared -Wl,-soname,libcolorprocesser.so -o libcolorprocesser.so colorprocesser.o -O3

