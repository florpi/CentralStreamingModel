#!/bin/bash

for box in {1..5}; do
	python first_stage.py halos $box
done


for box in {1..5}; do
	python first_stage.py galaxies $box
done
