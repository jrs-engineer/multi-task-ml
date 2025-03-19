import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# Define the Multi-Task Model
class MultiTaskModel(nn.Module):
    # kwargs: additional arguments for the model
    # freeze_backbone: whether to freeze the BERT backbone
    # freeze_task_a: whether to freeze the classification head
    # freeze_task_b: whether to freeze the NER head
    # use_qlora: whether to use QLoRA quantization
    # lora_r: row rank
    # lora_alpha: scaling factor
    # lora_dropout: dropout probability
    def __init__(self, num_classes_task_a, num_labels_task_b, **kwargs):   
        super(MultiTaskModel, self).__init__()
        if kwargs.get('use_qlora', False):
            # Use QLoRA quantization
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            bert = BertModel.from_pretrained('bert-base-uncased', quantization_config=quantization_config)
            # Define LoRA configuration
            lora_config = LoraConfig(
                r=kwargs.get('lora_r', 8),                          # Rank of the low-rank matrices
                lora_alpha=kwargs.get('lora_alpha', 32),            # Scaling factor
                target_modules=["query", "key", "value", "dense"],  # Apply LoRA to attention layers
                lora_dropout=kwargs.get('lora_dropout', 0.1),       # Dropout for regularization
                bias="none"                                         # No bias adaptation
            )
            self.bert = get_peft_model(prepare_model_for_kbit_training(bert), peft_config=lora_config)
        else:
            self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Task A: Classification head
        self.classification_head = nn.Linear(768, num_classes_task_a)
        # Task B: NER head
        self.ner_head = nn.Linear(768, num_labels_task_b)

        # Apply freeze options from config
        if kwargs.get('freeze_backbone', False) and not kwargs.get('use_qlora', False):
            # Freeze the BERT backbone
            for param in self.bert.parameters():
                param.requires_grad = False
        if kwargs.get('freeze_task_a', False):
            for param in self.classification_head.parameters():
                param.requires_grad = False
        if kwargs.get('freeze_task_b', False):
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