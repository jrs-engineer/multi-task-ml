import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

# Define the Multi-Task Model
class MultiTaskModel(nn.Module):
    def __init__(self, num_classes_task_a, num_labels_task_b,
                 freeze_backbone=False, freeze_task_a=False, freeze_task_b=False):
        super(MultiTaskModel, self).__init__()
        # Load pre-trained BERT as the backbone
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Task A: Classification head
        self.classification_head = nn.Linear(768, num_classes_task_a)
        # Task B: NER head
        self.ner_head = nn.Linear(768, num_labels_task_b)

        # Apply freeze options from config
        if freeze_backbone:
            for param in self.bert.parameters():
                param.requires_grad = False
        if freeze_task_a:
            for param in self.classification_head.parameters():
                param.requires_grad = False
        if freeze_task_b:
            for param in self.ner_head.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        # Forward pass through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        cls_embedding = last_hidden_states[:, 0, :]  # [CLS] token for classification
        # Task A: Sentence-level classification
        sentence_logits = self.classification_head(cls_embedding)
        # Task B: Token-level NER
        token_logits = self.ner_head(last_hidden_states)
        return sentence_logits, token_logits

# Test the multi-task model
def test_multi_task_model():
    # Determine device (CPU or CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = MultiTaskModel(num_classes_task_a=3, num_labels_task_b=4)
    model.to(device)
    
    # Sample input
    sentences = ["Hello, world!", "This is a test."]
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # Forward pass
    with torch.no_grad():
        sentence_logits, token_logits = model(inputs['input_ids'].to(device), inputs['attention_mask'].to(device))
    
    # Print shapes
    print(f"Sentence logits shape: {sentence_logits.shape}")  # Expected: [2, 3]
    print(f"Token logits shape: {token_logits.shape}")       # Expected: [2, seq_length, 4]

if __name__ == "__main__":
    test_multi_task_model()