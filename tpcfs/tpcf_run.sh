#!/bin/bash

for box in {101..116}; do
	python measure_tpcfs.py $box
done
