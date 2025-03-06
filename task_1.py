import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

# Define the Sentence Transformer class
class SentenceTransformer(nn.Module):
    def __init__(self):
        super(SentenceTransformer, self).__init__()
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Extract [CLS] token embedding (first token of the last hidden state)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # Shape: [batch_size, 768]
        return cls_embedding

# Test the implementation
def test_sentence_transformer():
    # Determine device (CPU or CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = SentenceTransformer()
    model.to(device)

    # Sample sentences
    sample_sentences = ["Hello, world!", "This is a test sentence."]

    # Tokenize inputs
    inputs = tokenizer(sample_sentences, return_tensors="pt", padding=True, truncation=True, max_length=128)

    # Move input tensors to the same device as the model
    inputs['input_ids'] = inputs['input_ids'].to(device)
    inputs['attention_mask'] = inputs['attention_mask'].to(device)

    # Get embeddings
    with torch.no_grad():  # Disable gradient computation for inference
        embeddings = model(inputs['input_ids'], inputs['attention_mask'])

    # Print results
    print(f"Embedding shape: {embeddings.shape}")  # Expected: [2, 768]
    print(f"Sample embedding (first 5 dimensions of first sentence): {embeddings[0, :5]}")

if __name__ == "__main__":
    test_sentence_transformer()