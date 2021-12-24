#!/bin/bash

g++ -lboost_system -c -fPIC polygoncalc.cpp -o polygoncalc.o  -O3  # -foptimize-sibling-calls  # -O1 -falign-functions  -falign-jumps \
#-falign-labels  -falign-loops \
#-fcaller-saves \
#-fcode-hoisting \
#-fcrossjumping \
#-fcse-follow-jumps  -fcse-skip-blocks \
#-fdelete-null-pointer-checks \
#-fdevirtualize  -fdevirtualize-speculatively \
#-fexpensive-optimizations \
#-fgcse  -fgcse-lm \
#-fhoist-adjacent-loads \
#-finline-functions \
#-finline-small-functions \
#-findirect-inlining \
#-fipa-bit-cp  -fipa-cp  -fipa-icf \
#-fipa-ra  -fipa-sra  -fipa-vrp \
#-fisolate-erroneous-paths-dereference \
#-fisolate-erroneous-paths-dereference \
#-flra-remat \
#-foptimize-sibling-call
#-foptimize-strlen \
#-fpartial-inlining \
#-fpeephole2 \
#-freorder-blocks-algorithm=stc
#-freorder-blocks-and-partition  -freorder-functions \
#-frerun-cse-after-loop  \
#-fschedule-insns  -fschedule-insns2 \
#-fsched-interblock  -fsched-spec \
#-fstore-merging \
#-fstrict-aliasing \
#-fthread-jumps \
#-ftree-builtin-call-dce \
#-ftree-pre \
#-ftree-switch-conversion  -ftree-tail-merge \
#-ftree-vrp


g++ -lboost_system -shared -Wl,-soname,libpolygoncalc.so -o libpolygoncalc.so polygoncalc.o -O3