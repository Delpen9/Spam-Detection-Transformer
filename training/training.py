import os
from transformers import BertTokenizerFast, DataCollatorForLanguageModeling, BertForMaskedLM
import torch


if __name__ == '__main__':
    # Path to the directory containing the text files
    directory_path = '../data/masking/openwebtext/openwebtext'

    # Load the BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # Initialize a list to hold the sentences
    sentences = []

    # Read all .txt files in the directory
    iteration = 0
    for filename in os.listdir(directory_path):
        iteration += 1
        if iteration > 1:
            break
        elif filename.endswith('.txt'):
            with open(os.path.join(directory_path, filename), 'r') as f:
                sentences.extend(f.read().strip().split('\n'))

    print('tokenizer')

    # Tokenize the sentences
    inputs = tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors='pt')

    # Convert inputs to list of tensors
    inputs = [{k: v[i].unsqueeze(0) for k, v in inputs.items()} for i in range(inputs["input_ids"].shape[0])]

    # Prepare the data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    print('mlm')

    # Generate the inputs for MLM
    mlm_inputs = [data_collator([input]) for input in inputs]

    # Initialize a BERT model for masked language modeling
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')

    # Move everything to the same device as the model
    mlm_inputs = {name: tensor.to(model.device) for name, tensor in mlm_inputs.items()}

    print('Forward pass.')

    # Perform a forward pass through the model
    outputs = model(**mlm_inputs)

    # The MLM loss can be accessed as follows
    loss = outputs.loss

    print(loss)