import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, BitsAndBytesConfig, AdamW
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
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
        max_length=256,  # Reduced max_length for efficiency
        return_tensors='pt'
    )

# Tokenize the dataset with batched processing
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=4,  # Use multiple processes for faster tokenization
    remove_columns=["text"]
)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# Split training data into train and validation sets (90% train, 10% validation)
train_val_dataset = tokenized_datasets['train'].train_test_split(test_size=0.1)
train_dataset = train_val_dataset['train']
val_dataset = train_val_dataset['test']
test_dataset = tokenized_datasets['test']

# Create data loaders with optimized batch size
batch_size = 32  # Increased batch size for better GPU utilization
num_workers = 8  # Increased workers for faster data loading
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

# Define quantization config for 4-bit QLoRA
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Load model in 4-bit precision
    bnb_4bit_quant_type="nf4",  # Use NF4 quantization (normalized float 4-bit)
    bnb_4bit_compute_dtype=torch.float16,  # Compute in float16 for efficiency
    bnb_4bit_use_double_quant=True  # Double quantization for additional memory savings
)

# Load BERT model with quantization
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=4,
    quantization_config=quant_config
)

# Prepare the model for 4-bit training
model = prepare_model_for_kbit_training(model)

# Define LoRA configuration for QLoRA
lora_config = LoraConfig(
    r=8,               # Lower rank for fewer parameters
    lora_alpha=16,     # Reduced scaling factor
    target_modules=["query", "value"],  # Focus on attention layers
    lora_dropout=0.1,  # Dropout for regularization
    bias="none",       # No bias adaptation
    task_type="SEQ_CLS" # Specify sequence classification task
)

# Apply LoRA to the quantized model
model = get_peft_model(model, lora_config)
model.to(device)

# Print trainable parameters for verification
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {trainable_params}")

# Set up optimizer with mixed precision
optimizer = AdamW(model.parameters(), lr=5e-4)

# Enable mixed precision training
scaler = torch.cuda.amp.GradScaler()

# Training loop with optimizations
num_epochs = 20
for epoch in range(num_epochs):
    print(f"\nTraining Epoch {epoch + 1}/{num_epochs}")
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        
        # Use autocast for mixed precision
        with torch.cuda.amp.autocast():
            outputs = model(**batch)
            loss = outputs.loss
        
        # Scale loss for mixed precision
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Average Training Loss: {avg_train_loss:.4f}")

    # Validation with no gradients
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

# Final evaluation on test set
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

# Save the model (QLoRA adapters and classification head)
model.save_pretrained("bert-qlora-agnews")