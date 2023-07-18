import os
import glob

# Define the directory where your text files are stored
folder_path = 'data/masking/openwebtext/openwebtext'

# Initialize maximum length to 0
max_length = 0
max_sentence = ''
filename_containing_sentence = ''

# Iterate over all .txt files in the directory
for filename in glob.glob(os.path.join(folder_path, '*.txt')):
    with open(filename, 'r') as file:
        # Read all sentences in each file
        sentences = file.readlines()
        # Iterate over sentences
        for sentence in sentences:
            # Check if current sentence is longer than the max_length
            if len(sentence) > max_length:
                max_length = len(sentence)
                max_sentence = sentence.strip()
                filename_containing_sentence = filename

# Print length of the longest sentence and the sentence itself
print(f'Length of the longest sentence: {max_length}')
print(f'Filename containing the longest sentence" {filename_containing_sentence}')