#!/bin/bash

# Directories to be cleared
declare -a dirs=("../artifacts/Fine-tuned" 
                 "../artifacts/MLM" 
                 "../output/illustrations" 
                 "../output/logging")

# Clear the directories
for dir in "${dirs[@]}"
do
    rm -r "$dir"/* 2>/dev/null
done

# Clear the files in the output directory
find "../output" -maxdepth 1 -type f -exec rm {} \;

echo "All specified directories and files in output directory have been cleared"