#!/bin/bash

folder="data/masking/openwebtext/openwebtext"

# Navigate to the specified folder
cd "$folder" || exit

# iterate over every file in the folder
for file in *.xz; do
    # De-compress the .xz files
    # This could potentially take a long time
    xz -d "$file"
done

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
#     tmp_file="temp_file.txt"

#     # remove extra whitespace lines from the .txt files
#     grep -E '.{100,}' "$file" > "$tmp_file"

#     mv "$tmp_file" "$file"
# done

# # iterate over each .txt file in the directory
# for file in *.txt; do
#     # cut each line to a maximum of 512 characters and overwrite the original file
#     cut -c -512 "${file}" > "${file}.tmp" && mv "${file}.tmp" "${file}"
# done

# # Find all .txt files in the directory
# for file in *.txt; do
#     # Use awk to remove the last word of each line and output to a temp file
#     awk '{$NF=""; print $0}' "$file" > "${file}.tmp"

#     # Overwrite the original file with the temp file
#     mv "${file}.tmp" "$file"
# done

# # Find all .txt files in the directory
# for file in *.txt; do
#     # Use tr to delete all punctuation from the file and output to a temp file
#     tr -d '[:punct:]' < "$file" > "${file}.tmp"

#     # Overwrite the original file with the temp file
#     mv "${file}.tmp" "$file"
# done