from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as.nn
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset
from transformers import BertTokenizer

class ReviewDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_len):
        self.data_file = open(data_file, 'r') 
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return sum(1 for line in open(self.data_file))

    def __getitem__(self, idx):
        line = next(itertools.islice(self.data_file, idx, None))
        review, target = line.strip().split(',') # assuming each line is "review,target"
        encoding = self.tokenizer.encode_plus(
          review,
          add_special_tokens=True,
          max_length=self.max_len,
          return_token_type_ids=False,
          pad_to_max_length=True,
          return_attention_mask=True,
          return_tensors='pt',
        )
        return {
          'review_text': review,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'targets': torch.tensor(target, dtype=torch.long)
        }

    def __del__(self):
        self.data_file.close()

# set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# initialize the model and tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device) # 2 labels: spam and not spam

# split your data into training and validation sets
train_reviews, val_reviews, train_targets, val_targets = train_test_split(reviews, targets, test_size=0.2)

# create dataloaders
BATCH_SIZE = 16
MAX_LEN = 256

train_dataset = ReviewDataset(train_reviews, train_targets, tokenizer, MAX_LEN)
val_dataset = ReviewDataset(val_reviews, val_targets, tokenizer, MAX_LEN)

train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# initialize the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss().to(device)

# train the model
EPOCHS = 4

for epoch in range(EPOCHS):
    model.train()
    for batch in train_data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['targets'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
