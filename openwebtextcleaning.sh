#!/bin/bash

folder="data/masking/openwebtext/openwebtext"

# Navigate to the specified folder
cd "$folder" || exit

# iterate over every file in the folder
for file in *; do
    # check if it's a file (not a directory)
    if [[ -f "$file" ]]; then
        # remove all non-ASCII characters and overwrite the file
        tr -cd '\11\12\15\40-\176' < "$file" > temp && mv temp "$file"
    fi
done

i=0
# iterate over every file in the folder
for file in *; do
    # check if it's a file (not a directory)
    if [[ -f "$file" ]]; then
        # rename files and provide .txt extention
        mv -- "$file" "batch_${i}.txt"
        ((i++))
    fi
done

# iterate over every file in the folder
for file in *.txt; do
    # check if it's a file (not a directory)
    if [[ -f "$file" ]]; then
        # remove extra whitespace lines from the .txt files
        sed -i '/^$/d' "$file"
    fi
done