# Multi-Task Model

## Task 1: Sentence Transformer Implementation

- **Model**: The SentenceTransformer class uses `bert-base-uncased` (768-dimensional embeddings) as the backbone. The [CLS] token embedding from BERT’s last hidden state is extracted as the sentence representation.
- **Why [CLS]?**: BERT is pre-trained to aggregate sentence-level information in the [CLS] token, making it ideal for tasks like classification. Alternatives like mean pooling could work for similarity tasks, but [CLS] is kept for consistency and simplicity.
- **Testing**: Two sample sentences are tokenized and passed through the model. The output shape [2, 768] confirms two fixed-length embeddings, each 768-dimensional.

```bash
pip install -r requirements.txt
python task_1.py

# Output:
Using device: cuda
Embedding shape: torch.Size([2, 768])
Sample embedding (first 5 dimensions of first sentence): tensor([-0.0781,  0.1587,  0.0400, -0.1986, -0.3442], device='cuda:0')
```

## Task 2: Multi-Task Learning Expansion

- Extend the Sentence Transformer model to support multi-task learning.
    - Task A: Sentence classification (e.g., "sports", "politics", "technology").
    - Task B: Named Entity Recognition (NER) (e.g., "PERSON", "ORG", "LOC", "O").

- **Architecture**:
    - Shared Backbone: BERT generates embeddings for both tasks.
    - Task A Head: A linear layer maps the [CLS] embedding to classification logits (e.g., 3 classes).
    - Task B Head: A linear layer maps each token’s embedding to NER logits (e.g., 4 labels).

- **Outputs**: Returns sentence_logits for Task A and token_logits for Task B, enabling simultaneous predictions.
- **Why NER?**: Chosen to demonstrate token-level predictions, contrasting with Task A’s sentence-level task, highlighting multi-task diversity.
- **Testing**: Verifies output shapes for a batch of two sentences.

```bash
python task_2.py

# Output:
Using device: cuda
Sentence logits shape: torch.Size([2, 3])
Token logits shape: torch.Size([2, 7, 4])
```

## Task 3: Training Considerations

### Freezing Scenarios

1. **Entire Network Frozen**

- Implications: No training occurs; the model is a fixed feature extractor.
- Advantages: None for training; useful for inference only.
- Use Case: Not suitable for learning new tasks.

2. **Transformer Backbone Frozen**

- Implications: Only task heads are trained; BERT features are static.
- Advantages:
    - Faster training (fewer parameters).
    - Less overfitting with small datasets.
    - Leverages pre-trained features.
- Use Case: Good for limited data or tasks close to BERT’s pre-training.

3. **One Task Head Frozen (e.g., Task A)**

- Implications: Backbone and Task B head train; Task A head is fixed.
- Advantages:
    - Preserves Task A performance while adapting to Task B.
    - Useful for adding new tasks incrementally.
- Use Case: Continual learning with pre-trained tasks.

### Transfer Learning Approach

- Process:
    - Start with `bert-base-uncased`, pre-trained on large corpora.
    - Fine-tune the entire model (backbone and heads) on Tasks A and B.
- Rationale:
    - Full fine-tuning adapts shared representations to both tasks.
    - Optimal with sufficient data; partial freezing (e.g., lower layers) could mitigate overfitting with limited data.
- Advantages: Maximizes performance by leveraging pre-trained weights and task-specific tuning.

## Task 4: Training Loop Implementation

- Assumptions:
    - A batch contains input_ids, attention_mask, labels_task_a (class IDs), and labels_task_b (NER labels, -100 for padding).
    - Simulated data is used; in practice, replace `ExampleDataset` with real data.
- Losses:
    - Task A: Cross-entropy for classification.
    - Task B: Cross-entropy for NER, ignoring padding tokens (-100).
    - Combined with equal weighting (1:1); tune weights in practice if needed.
- Optimizer: AdamW with lr=2e-5, standard for BERT fine-tuning. It is configurable in the config file.

To make `Multi-Task Learning` model configurable, you can use a config.yml file to define all hyperparameters (like learning rate) and freeze options (e.g., freezing the backbone or specific task heads).

```yaml
model:
  num_classes_task_a: 3        # Number of classes for Task A (e.g., classification)
  num_labels_task_b: 4         # Number of labels for Task B (e.g., NER)
  freeze_backbone: true        # Freeze the BERT backbone (true/false)

training:
  learning_rate: 2e-5          # Learning rate for the optimizer
  num_epochs: 10               # Number of training epochs
  batch_size: 2                # Batch size for DataLoader
  task: both                   # Which task(s) to train: 'task_a', 'task_b', or 'both'

device: cuda                   # Device to use: 'cuda' or 'cpu'
```

**Training Loop**: Iterates over the dataset, computes losses for both tasks, and backpropagates gradients.

```bash
python task_4.py

# Output:
Epoch 1/10, Loss: 2.3549392223358154
Epoch 2/10, Loss: 2.341250419616699
Epoch 3/10, Loss: 2.357252597808838
Epoch 4/10, Loss: 2.4723000526428223
Epoch 5/10, Loss: 2.269099235534668
Epoch 6/10, Loss: 2.4188179969787598
Epoch 7/10, Loss: 2.2739875316619873
Epoch 8/10, Loss: 2.250629186630249
Epoch 9/10, Loss: 2.314945936203003
Epoch 10/10, Loss: 2.3897457122802734
```
