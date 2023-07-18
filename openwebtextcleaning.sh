#!/bin/bash

folder="data/masking/openwebtext/openwebtext"

# Navigate to the specified folder
cd "$folder"

# iterate over every file in the folder
for file in *; do
    # check if it's a file (not a directory)
    if [[ -f "$file" ]]; then
        # remove all non-ASCII characters and overwrite the file
        tr -cd '\11\12\15\40-\176' < "$file" > temp && mv temp "$file"
    fi
done