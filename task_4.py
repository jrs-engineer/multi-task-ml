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

def evaluate_model(model, test_loader, task, device):
    model.eval()
    correct_a = 0
    total_a = 0
    correct_b = 0
    total_b = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_a = batch['labels_a'].to(device)
            labels_b = batch['labels_b'].to(device)

            sentence_logits, token_logits = model(input_ids, attention_mask)

            # Evaluate task_a
            if task in ['task_a', 'both']:
                pred_a = sentence_logits.argmax(dim=1)
                correct_a += (pred_a == labels_a).sum().item()
                total_a += len(labels_a)

            # Evaluate task_b
            if task in ['task_b', 'both']:
                flat_token_logits = token_logits.view(-1, token_logits.size(-1))
                flat_labels_b = labels_b.view(-1)
                valid_positions = flat_labels_b != -100
                flat_labels_b_valid = flat_labels_b[valid_positions]
                flat_token_logits_valid = flat_token_logits[valid_positions]
                pred_b = flat_token_logits_valid.argmax(dim=1)
                correct_b += (pred_b == flat_labels_b_valid).sum().item()
                total_b += len(flat_labels_b_valid)

    # Calculate accuracies
    acc_a = correct_a / total_a if total_a > 0 else 0
    acc_b = correct_b / total_b if total_b > 0 else 0

    print(f"Accuracy for task_a: {acc_a:.4f}")
    print(f"Accuracy for task_b: {acc_b:.4f}")
    
def train_model(model, optimizer, train_loader, num_epochs, task, device):
    for epoch in range(num_epochs):
        model.train()
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
        evaluate_model(model, train_loader, task, device)
        

# Example dataset (replace with real data)
class ExampleDataset:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.sentences = ["I love sports.", "Jack and Jill went up the hill.", "Google is in tech."]
        self.labels_a = [1, 0, 3]
        self.labels_b = [[-100, 0, 0, 0, 0, -100, -100, -100], [-100, 1, 0, 1, 0, 0, 0, -100], [-100, 2, 0, 0, 0, 0, -100, -100]]

    def __len__(self): return len(self.sentences)
    def __getitem__(self, idx):
        encoding = self.tokenizer(self.sentences[idx], return_tensors='pt', padding='max_length', max_length=8, truncation=True)
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