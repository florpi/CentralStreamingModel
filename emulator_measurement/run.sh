#!/bin/bash

for box in {101..115}; do
	python first_stage.py $box
done
