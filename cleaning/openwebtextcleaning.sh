#!/bin/bash

folder="../data/masking/openwebtext/openwebtext"

# Navigate to the specified folder
cd "$folder" || exit

# # iterate over every file in the folder
# for file in *.xz; do
#     # De-compress the .xz files
#     # This could potentially take a long time
#     xz -d "$file"
# done

# # iterate over every file in the folder
# for file in *; do
#     # remove all non-ASCII characters and overwrite the file
#     tr -cd '\11\12\15\40-\176' < "$file" > temp && mv temp "$file"
# done

# i=0
# # iterate over every file in the folder
# for file in *; do
#     # rename files and provide .txt extension
#     mv -- "$file" "batch_${i}.txt"
#     ((i++))
# done

# # iterate over every file in the folder
# for file in *.txt; do
#     # remove extra whitespace lines from the .txt files
#     sed -i '/^$/d' "$file"
# done

# # iterate over every file in the folder
# for file in *.txt; do
#     # Create temp file
#     tmp_file="${file}.tmp"

#     # remove lines from .txt with less than 100 characters
#     grep -E '.{100,}' "$file" > "$tmp_file"

#     mv "$tmp_file" "$file"
# done

# # iterate over every file in the folder
# for file in *.txt; do
#     # Use head to get the first 4160 lines and save them to a temp file
#     head -n 4160 "$file" > "${file}.tmp"

#     # Replace the original file with the temp file
#     mv "${file}.tmp" "$file"
# done

# min_lines=4160
# for file in *.txt; do
#     if [ $(wc -l <"$file") -lt $min_lines ]; then
#         rm "$file"
#     fi
# done