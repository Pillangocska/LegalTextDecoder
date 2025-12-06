from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.optim import AdamW
from pathlib import Path
from tqdm import tqdm
import seaborn as sns
import torch.nn as nn
import pandas as pd
import numpy as np
import warnings
import datetime
import pickle
import torch
import yaml

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# =============================================================================
# CONFIGURATION
# =============================================================================
model_name = 'SZTAKI-HLT/hubert-base-cc'
tokenizer_model_name = 'SZTAKI-HLT/hubert-base-cc'
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
NUM_EPOCHS = 10
ACCUMULATION_STEPS = 2
MAX_LENGTH = 256
EARLY_STOPPING_PATIENCE = 3

# Regularization
HIDDEN_DROPOUT = 0.2  # Slightly reduced from 0.3
ATTENTION_DROPOUT = 0.2
FREEZE_LAYERS = 4  # Reduced from 6 - allow more fine-tuning

# NEW: Sampling balance factor (0 = no balancing, 1 = full inverse frequency)
# Using sqrt balancing as a middle ground
SAMPLING_BALANCE_POWER = 0.5  # Square root of inverse frequency

print("=" * 80)
print("IMPROVED ÁSZF READABILITY MODEL TRAINING - V2")
print("=" * 80)
print("\nKey changes from V1:")
print("  1. SOFT weighted sampling (sqrt of inverse frequency)")
print("  2. NO class weights in loss (sampling handles balance)")
print("  3. Ordinal loss without class weights")
print("  4. Reduced frozen layers (4 instead of 6)")
print("  5. Slightly lower dropout (0.2 instead of 0.3)")


# =============================================================================
# ORDINAL-AWARE LOSS (WITHOUT CLASS WEIGHTS)
# =============================================================================
class OrdinalCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss with ordinal penalty.
    No class weights - sampling handles class balance.
    """
    def __init__(self, num_classes=5, ordinal_lambda=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.ordinal_lambda = ordinal_lambda

    def forward(self, logits, targets):
        # Standard cross-entropy WITHOUT class weights
        ce_loss = F.cross_entropy(logits, targets)

        # Ordinal penalty
        probs = F.softmax(logits, dim=1)
        classes = torch.arange(self.num_classes, device=logits.device).float()
        predicted_means = (probs * classes).sum(dim=1)
        ordinal_penalty = (predicted_means - targets.float()).pow(2).mean()

        return ce_loss + self.ordinal_lambda * ordinal_penalty


# =============================================================================
# DATA PREPROCESSING
# =============================================================================
print("\n" + "=" * 80)
print("DATA PREPROCESSING")
print("=" * 80)

train_path = Path('/content/sample_data/train.csv')
test_path = Path('/content/sample_data/test.csv')

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

print(f"\nTraining data: {df_train.shape}")
print(f"Test data: {df_test.shape}")

tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
print(f"✓ Tokenizer loaded (vocab size: {tokenizer.vocab_size})")


# =============================================================================
# TOKENIZATION
# =============================================================================
print("\n" + "=" * 80)
print("TOKENIZING DATASETS")
print("=" * 80)

def tokenize_texts(texts, labels, tokenizer, max_length):
    encodings = tokenizer(
        texts.tolist(),
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors=None
    )
    return {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels.tolist()
    }

train_encodings = tokenize_texts(
    df_train['text'].values,
    df_train['label_numeric'].values,
    tokenizer,
    MAX_LENGTH
)
print(f"✓ Training data tokenized: {len(train_encodings['input_ids'])} samples")

test_encodings = tokenize_texts(
    df_test['text'].values,
    df_test['label_numeric'].values,
    tokenizer,
    MAX_LENGTH
)
print(f"✓ Test data tokenized: {len(test_encodings['input_ids'])} samples")


# =============================================================================
# TRAIN/VALIDATION SPLIT
# =============================================================================
print("\n" + "=" * 80)
print("CREATING TRAIN/VALIDATION SPLIT")
print("=" * 80)

train_indices, val_indices = train_test_split(
    range(len(train_encodings['input_ids'])),
    test_size=0.2,
    random_state=42,
    stratify=train_encodings['labels']
)

print(f"\nSplit sizes:")
print(f"  Training:   {len(train_indices)} samples")
print(f"  Validation: {len(val_indices)} samples")
print(f"  Test:       {len(test_encodings['input_ids'])} samples")

train_dataset = {
    'input_ids': [train_encodings['input_ids'][i] for i in train_indices],
    'attention_mask': [train_encodings['attention_mask'][i] for i in train_indices],
    'labels': [train_encodings['labels'][i] for i in train_indices]
}

val_dataset = {
    'input_ids': [train_encodings['input_ids'][i] for i in val_indices],
    'attention_mask': [train_encodings['attention_mask'][i] for i in val_indices],
    'labels': [train_encodings['labels'][i] for i in val_indices]
}

# Label distribution
print("\nLabel distribution in training:")
train_labels_dist = pd.Series(train_dataset['labels']).value_counts().sort_index()
for label, count in train_labels_dist.items():
    pct = (count / len(train_dataset['labels'])) * 100
    print(f"  Class {label}: {count:4d} samples ({pct:5.2f}%)")


# =============================================================================
# SAVE PREPROCESSED DATA
# =============================================================================
output_dir = Path('../_data/preprocessed')
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / 'train_dataset.pkl', 'wb') as f:
    pickle.dump(train_dataset, f)
with open(output_dir / 'val_dataset.pkl', 'wb') as f:
    pickle.dump(val_dataset, f)
with open(output_dir / 'test_dataset.pkl', 'wb') as f:
    pickle.dump(test_encodings, f)

config = {
    'model_name': tokenizer_model_name,
    'max_length': MAX_LENGTH,
    'train_size': len(train_dataset['labels']),
    'val_size': len(val_dataset['labels']),
    'test_size': len(test_encodings['labels']),
    'num_classes': 5
}

with open(output_dir / 'config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

print(f"\n✓ Preprocessed data saved to: {output_dir}")


# =============================================================================
# MODEL TRAINING SETUP
# =============================================================================
print("\n" + "=" * 80)
print("MODEL TRAINING SETUP")
print("=" * 80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

torch.manual_seed(42)
np.random.seed(42)


class ASZFDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['labels'])

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx], dtype=torch.long),
            'labels': torch.tensor(self.encodings['labels'][idx] - 1, dtype=torch.long)
        }


train_torch_dataset = ASZFDataset(train_dataset)
val_torch_dataset = ASZFDataset(val_dataset)
test_torch_dataset = ASZFDataset(test_encodings)


# =============================================================================
# SOFT WEIGHTED SAMPLER
# =============================================================================
print("\n" + "=" * 80)
print("CREATING SOFT WEIGHTED SAMPLER")
print("=" * 80)

train_labels_original = np.array(train_dataset['labels'])
class_counts = np.bincount(train_labels_original)[1:]  # Skip index 0

# SOFT weighting: use power < 1 for gentler balancing
# power=0.5 means sqrt of inverse frequency
# power=1.0 would be full inverse frequency (aggressive)
# power=0.0 would be uniform (no balancing)
class_weights_for_sampling = (1.0 / class_counts) ** SAMPLING_BALANCE_POWER

# Normalize so weights sum to num_classes (keeps effective sample size similar)
class_weights_for_sampling = class_weights_for_sampling / class_weights_for_sampling.sum() * 5

sample_weights = class_weights_for_sampling[train_labels_original - 1]

print(f"\nSampling balance power: {SAMPLING_BALANCE_POWER} (0=none, 1=full inverse)")
print("\nClass sampling weights (soft balanced):")
for i, (count, weight) in enumerate(zip(class_counts, class_weights_for_sampling)):
    effective_prob = weight / class_weights_for_sampling.sum()
    print(f"  Class {i+1}: {count:4d} samples, weight={weight:.4f}, effective_prob={effective_prob*100:.1f}%")

# Compare to original distribution
print("\nExpected class distribution per epoch:")
print("  Original | Soft balanced | Full balanced")
for i, count in enumerate(class_counts):
    orig_pct = count / class_counts.sum() * 100
    soft_pct = class_weights_for_sampling[i] / class_weights_for_sampling.sum() * 100
    full_pct = 20.0  # With full balancing, each class would be 20%
    print(f"  Class {i+1}: {orig_pct:5.1f}%  |    {soft_pct:5.1f}%     |    {full_pct:5.1f}%")

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

print("\n✓ Soft weighted sampler created")


# =============================================================================
# DATALOADERS
# =============================================================================
train_loader = DataLoader(
    train_torch_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler
)

val_loader = DataLoader(
    val_torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"\n✓ DataLoaders created")
print(f"  Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * ACCUMULATION_STEPS})")
print(f"  Learning rate: {LEARNING_RATE}")


# =============================================================================
# MODEL WITH MODERATE REGULARIZATION
# =============================================================================
print("\n" + "=" * 80)
print(f"LOADING MODEL: {model_name}")
print("=" * 80)

num_labels = config['num_classes']

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    hidden_dropout_prob=HIDDEN_DROPOUT,
    attention_probs_dropout_prob=ATTENTION_DROPOUT,
    problem_type="single_label_classification"
)

# Freeze early layers
print(f"\nFreezing first {FREEZE_LAYERS} transformer layers...")

for param in model.bert.embeddings.parameters():
    param.requires_grad = False

for layer_idx in range(FREEZE_LAYERS):
    for param in model.bert.encoder.layer[layer_idx].parameters():
        param.requires_grad = False

model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params

print(f"\n✓ Model loaded to {device}")
print(f"  Total parameters:     {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Frozen parameters:    {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
print(f"  Dropout: {HIDDEN_DROPOUT}")


# =============================================================================
# LOSS FUNCTION (NO CLASS WEIGHTS - SAMPLER HANDLES BALANCE)
# =============================================================================
print("\n" + "=" * 80)
print("LOSS FUNCTION: Ordinal Cross-Entropy (no class weights)")
print("=" * 80)

criterion = OrdinalCrossEntropyLoss(
    num_classes=5,
    ordinal_lambda=0.5
)

print(f"\n✓ OrdinalCrossEntropyLoss initialized")
print(f"  NO class weights (sampler handles balance)")
print(f"  Ordinal lambda: 0.5")


# =============================================================================
# OPTIMIZER AND SCHEDULER
# =============================================================================
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {
        'params': [p for n, p in model.named_parameters()
                   if not any(nd in n for nd in no_decay) and p.requires_grad],
        'weight_decay': 0.01
    },
    {
        'params': [p for n, p in model.named_parameters()
                   if any(nd in n for nd in no_decay) and p.requires_grad],
        'weight_decay': 0.0
    }
]

optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)

total_steps = len(train_loader) * NUM_EPOCHS // ACCUMULATION_STEPS
warmup_steps = int(0.1 * total_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

print(f"\nOptimizer: AdamW (lr={LEARNING_RATE}, weight_decay=0.01)")
print(f"Scheduler: Linear warmup ({warmup_steps} steps) + decay")


# =============================================================================
# LOGGING
# =============================================================================
log_dir = Path('../logs')
log_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = log_dir / f'training_v2_{timestamp}.txt'

with open(log_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("IMPROVED ÁSZF MODEL TRAINING - V2\n")
    f.write("=" * 80 + "\n")
    f.write(f"Model: {model_name}\n")
    f.write(f"Trainable params: {trainable_params:,}\n")
    f.write(f"Frozen layers: {FREEZE_LAYERS}\n")
    f.write(f"Dropout: {HIDDEN_DROPOUT}\n")
    f.write(f"Learning rate: {LEARNING_RATE}\n")
    f.write(f"Sampling balance power: {SAMPLING_BALANCE_POWER}\n")
    f.write(f"Loss: OrdinalCE (no class weights, lambda=0.5)\n")
    f.write("=" * 80 + "\n\n")

print(f"\n✓ Logging to: {log_file}")


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================
def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, epoch, log_file, accum_steps):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    optimizer.zero_grad()

    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        loss = criterion(logits, labels) / accum_steps
        loss.backward()

        if (batch_idx + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item() * accum_steps

        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        progress_bar.set_postfix({
            'loss': f'{loss.item() * accum_steps:.4f}',
            'acc': f'{correct/total:.4f}'
        })

    if len(dataloader) % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    # Show per-class accuracy during training
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print(f"\n  Per-class recall (train):", end=" ")
    for c in range(5):
        mask = all_labels == c
        if mask.sum() > 0:
            recall = (all_preds[mask] == c).sum() / mask.sum()
            print(f"C{c+1}:{recall:.2f}", end=" ")
    print()

    with open(log_file, 'a') as f:
        f.write(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}, Train Acc: {accuracy:.4f}\n")

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device, split_name, log_file):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'Eval {split_name}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)

            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    with open(log_file, 'a') as f:
        f.write(f"{split_name} - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}\n")

    return avg_loss, accuracy, np.array(all_predictions), np.array(all_labels)


# =============================================================================
# TRAINING LOOP
# =============================================================================
print("\n" + "=" * 80)
print("STARTING TRAINING")
print("=" * 80)

best_val_loss = float('inf')
best_val_accuracy = 0.0
best_epoch = 0
patience_counter = 0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

model_dir = Path('../models')
model_dir.mkdir(parents=True, exist_ok=True)
best_model_path = model_dir / 'model_best_v2.pt'

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\n{'=' * 80}")
    print(f"EPOCH {epoch}/{NUM_EPOCHS}")
    print(f"{'=' * 80}")

    train_loss, train_acc = train_epoch(
        model, train_loader, optimizer, scheduler, criterion,
        device, epoch, log_file, ACCUMULATION_STEPS
    )
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

    val_loss, val_acc, val_preds, val_labels = evaluate(
        model, val_loader, criterion, device, f"Val (Epoch {epoch})", log_file
    )
    print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")

    # Show per-class recall on validation
    print(f"  Per-class recall (val):", end=" ")
    for c in range(5):
        mask = val_labels == c
        if mask.sum() > 0:
            recall = (val_preds[mask] == c).sum() / mask.sum()
            print(f"C{c+1}:{recall:.2f}", end=" ")
    print()

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_accuracy = val_acc
        best_epoch = epoch
        patience_counter = 0

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'val_accuracy': val_acc,
        }, best_model_path)

        print(f"✓ Best model saved! (Val Loss: {val_loss:.4f})")
    else:
        patience_counter += 1
        print(f"  No improvement. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n⚠ Early stopping triggered at epoch {epoch}")
            break

print(f"\n{'=' * 80}")
print("TRAINING COMPLETE!")
print(f"Best validation loss: {best_val_loss:.4f} (Epoch {best_epoch})")
print(f"Best validation accuracy: {best_val_accuracy:.4f}")


# =============================================================================
# FINAL EVALUATION
# =============================================================================
print("\n" + "=" * 80)
print("FINAL EVALUATION")
print("=" * 80)

checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"\n✓ Loaded best model from epoch {checkpoint['epoch']}")

val_loss, val_acc, val_preds, val_labels = evaluate(
    model, val_loader, criterion, device, "Final Val", log_file
)

print("\nClassification Report (Validation Set):")
print(classification_report(
    val_labels, val_preds,
    target_names=[f'Class {i}' for i in range(1, 6)],
    digits=4
))

cm = confusion_matrix(val_labels, val_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f'Pred {i}' for i in range(1, 6)],
            yticklabels=[f'True {i}' for i in range(1, 6)])
plt.title('Confusion Matrix (Validation Set) - V2', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(model_dir / 'confusion_matrix_v2.png', dpi=150)
plt.show()

# Training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
epochs_range = range(1, len(history['train_loss']) + 1)

axes[0].plot(epochs_range, history['train_loss'], 'b-o', label='Train Loss')
axes[0].plot(epochs_range, history['val_loss'], 'r-o', label='Val Loss')
axes[0].axvline(x=best_epoch, color='g', linestyle='--', label=f'Best (Epoch {best_epoch})')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(epochs_range, history['train_acc'], 'b-o', label='Train Accuracy')
axes[1].plot(epochs_range, history['val_acc'], 'r-o', label='Val Accuracy')
axes[1].axvline(x=best_epoch, color='g', linestyle='--', label=f'Best (Epoch {best_epoch})')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training and Validation Accuracy')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(model_dir / 'training_history_v2.png', dpi=150)
plt.show()


# =============================================================================
# TEST SET EVALUATION
# =============================================================================
print("\n" + "=" * 80)
print("TEST SET EVALUATION")
print("=" * 80)

test_loss, test_acc, test_preds, test_labels = evaluate(
    model, test_loader, criterion, device, "Test", log_file
)

print("\nClassification Report (Test Set):")
print(classification_report(
    test_labels, test_preds,
    target_names=[f'Class {i}' for i in range(1, 6)],
    digits=4
))

cm_test = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens',
            xticklabels=[f'Pred {i}' for i in range(1, 6)],
            yticklabels=[f'True {i}' for i in range(1, 6)])
plt.title('Confusion Matrix (Test Set) - V2', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(model_dir / 'confusion_matrix_test_v2.png', dpi=150)
plt.show()


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("TRAINING SUMMARY - V2")
print("=" * 80)

print(f"""
Model: {model_name}

Configuration:
  ✓ Soft weighted sampling (power={SAMPLING_BALANCE_POWER})
  ✓ Ordinal loss WITHOUT class weights (lambda=0.5)
  ✓ Dropout: {HIDDEN_DROPOUT}
  ✓ Frozen layers: {FREEZE_LAYERS}
  ✓ Learning rate: {LEARNING_RATE}
  ✓ Early stopping (patience={EARLY_STOPPING_PATIENCE})

Results:
  Best epoch: {best_epoch}
  Best validation loss: {best_val_loss:.4f}
  Best validation accuracy: {best_val_accuracy:.4f}
  Test accuracy: {test_acc:.4f}

Files saved:
  Model: {best_model_path.absolute()}
  Log: {log_file.absolute()}
  Plots: {model_dir.absolute()}
""")

print("SCRIPT COMPLETE!")
