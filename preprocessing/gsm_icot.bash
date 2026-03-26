#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# Create data directory if it doesn't exist
mkdir -p data
# Download and process GSM8K dataset for Internalize CoT


for split in train valid test; do
  python preprocessing/gsm_icot.py ${split}
  rm data/gsm_${split}.txt
done