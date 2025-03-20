import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the AG News dataset
dataset = load_dataset("ag_news")

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define efficient tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=256,
        return_tensors='pt'
    )

# Tokenize the dataset with batched processing
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=4,
    remove_columns=["text"]
)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# Split training data
train_val_dataset = tokenized_datasets['train'].train_test_split(test_size=0.1)
train_dataset = train_val_dataset['train']
val_dataset = train_val_dataset['test']
test_dataset = tokenized_datasets['test']

# Create data loaders
batch_size = 32
num_workers = 8
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

# Load standard BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
model.to(device)

# Print all parameters count
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {total_params}")  # All parameters are trainable

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)  # Lower learning rate recommended for full fine-tuning

# Training loop
num_epochs = 10  # Fewer epochs recommended for full fine-tuning
for epoch in range(num_epochs):
    print(f"\nTraining Epoch {epoch + 1}/{num_epochs}")
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Average Training Loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    total_eval_accuracy = 0
    for batch in tqdm(val_dataloader, desc="Validation"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        total_eval_accuracy += (predictions == batch['labels']).float().mean().item()
    avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
    print(f"Validation Accuracy: {avg_val_accuracy:.4f}")

# Final evaluation
model.eval()
total_test_accuracy = 0
for batch in tqdm(test_dataloader, desc="Testing"):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    total_test_accuracy += (predictions == batch['labels']).float().mean().item()
avg_test_accuracy = total_test_accuracy / len(test_dataloader)
print(f"\nTest Accuracy: {avg_test_accuracy:.4f}")

# Save the full model
model.save_pretrained("bert-full-finetuned-agnews")