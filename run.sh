#!/bin/bash

./build.sh

export OMP_NUM_THREADS=8
export OMP_PLACES=cores
export OMP_PROC_BIND=close

./build/main --dataset nq --task task3 --params clusters