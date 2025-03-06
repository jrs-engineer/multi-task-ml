import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer
import yaml
from task_2 import MultiTaskModel

# Load config
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

def train_model(model, optimizer, train_loader, num_epochs, task, device):
    model.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_a = batch['labels_a'].to(device)  # Task A labels
            labels_b = batch['labels_b'].to(device)  # Task B labels

            optimizer.zero_grad()
            sentence_logits, token_logits = model(input_ids, attention_mask)

            # Compute loss based on the task specified in config
            if task == 'task_a':
                loss = nn.CrossEntropyLoss()(sentence_logits, labels_a)
            elif task == 'task_b':
                loss = nn.CrossEntropyLoss(ignore_index=-100)(token_logits.view(-1, token_logits.size(-1)), labels_b.view(-1))
            else:  # 'both'
                loss_a = nn.CrossEntropyLoss()(sentence_logits, labels_a)
                loss_b = nn.CrossEntropyLoss(ignore_index=-100)(token_logits.view(-1, token_logits.size(-1)), labels_b.view(-1))
                loss = loss_a + loss_b

            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
        

# Example dataset (replace with real data)
class ExampleDataset:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.sentences = ["I love sports.", "Google is in tech."]
        self.labels_a = [0, 2]
        self.labels_b = [[0, 0, 0, -100], [1, 0, 2, -100]]

    def __len__(self): return len(self.sentences)
    def __getitem__(self, idx):
        encoding = self.tokenizer(self.sentences[idx], return_tensors='pt', padding='max_length', max_length=4, truncation=True)
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels_a': torch.tensor(self.labels_a[idx]),
            'labels_b': torch.tensor(self.labels_b[idx])
        }
        
# Setup
dataset = ExampleDataset()
train_loader = DataLoader(dataset, batch_size=config['training']['batch_size'])
model = MultiTaskModel(
    num_classes_task_a=config['model']['num_classes_task_a'],
    num_labels_task_b=config['model']['num_labels_task_b'],
    freeze_backbone=config['model']['freeze_backbone'],
    freeze_task_a=config['training']['task']=='task_b',
    freeze_task_b=config['training']['task']=='task_a'
)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=float(config['training']['learning_rate']))
device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
model.to(device)

# Train
train_model(model, optimizer, train_loader, config['training']['num_epochs'], config['training']['task'], device)