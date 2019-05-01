#!/bin/bash

DEBUG=""

if [ $# == 1 ]
then
    if [ $1 == "debug" ]
    then
	DEBUG="--mca plm_base_verbose 10"
    fi
fi

mpiexec $DEBUG --hostfile ./hostfile python3 ~/CodedElasticComputingImplementation/main.py
