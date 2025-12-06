# %%
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from sklearn.dummy import DummyClassifier
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load training data
train_path = Path('../_data/final/train.csv')
df_train = pd.read_csv(train_path)

print(f"\nTraining data loaded: {df_train.shape}")
print(f"Columns: {df_train.columns.tolist()}")

print("\nLabel distribution:")
print(df_train['label_numeric'].value_counts().sort_index())

# Prepare features and labels
X = df_train['text'].values
y = df_train['label_numeric'].values

print(f"\nFeatures (X): {X.shape}")
print(f"Labels (y): {y.shape}")
print(f"Unique labels: {np.unique(y)}")

# %%
# Load test data
test_path = Path('../_data/final/test.csv')
df_test = pd.read_csv(test_path)

print(f"\nTest data loaded: {df_test.shape}")

X_test = df_test['text'].values
y_test = df_test['label_numeric'].values

print("\nDataset sizes:")
print(f"  Train: {len(X)} samples")
print(f"  Test:  {len(X_test)} samples")

print("\nTest label distribution:")
test_label_dist = pd.Series(y_test).value_counts().sort_index()
for label, count in test_label_dist.items():
    pct = (count / len(y_test)) * 100
    print(f"  Label {label}: {count:3d} ({pct:5.2f}%)")

# %%
# Train dummy classifier (most frequent strategy)
dummy_clf = DummyClassifier(strategy='most_frequent', random_state=42)
dummy_clf.fit(X, y)

# Predict on test set
y_pred_dummy = dummy_clf.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred_dummy)
f1_macro = f1_score(y_test, y_pred_dummy, average='macro')
f1_weighted = f1_score(y_test, y_pred_dummy, average='weighted')

print(f"\nMost frequent class in training: {dummy_clf.classes_[np.argmax(dummy_clf.class_prior_)]}")
print(f"Frequency: {np.max(dummy_clf.class_prior_):.2%}")

print(f"\nAccuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"F1-Macro:     {f1_macro:.4f}")
print(f"F1-Weighted:  {f1_weighted:.4f}")

print(classification_report(y_test, y_pred_dummy, target_names=[f'Label {i}' for i in range(1, 6)]))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_dummy)
print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f'Pred {i}' for i in range(1, 6)],
            yticklabels=[f'True {i}' for i in range(1, 6)])
plt.title('Confusion Matrix - Most Frequent Class Baseline', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.show()

print(f"\nThis baseline always predicts class {dummy_clf.classes_[np.argmax(dummy_clf.class_prior_)]}")
print(f"Achieves {accuracy*100:.2f}% accuracy by simply guessing the most common class")
print("\nAny model we build MUST beat this baseline to be useful!")

# %% [markdown]
# # Baseline Model: DummyClassifier (Most Frequent Strategy)
#
# Approach: The baseline model always predicts class 4 (Érthető - Understandable), which is the most common label in the training data, appearing in 31.83% of training samples.
#
# Performance on Test Set:
# - Accuracy: 20.45% (27 out of 132 predictions correct)
# - F1-Macro Score: 0.0679
# - F1-Weighted Score: 0.0695
#
# The model exclusively predicts class 4, completely ignoring all other readability levels (classes 1, 2, 3, and 5). This results in 0% precision and recall for all non-majority classes. The 20.45% accuracy reflects the proportion of class 4 samples in the test set. This is actually lower than the training distribution (31.83%), indicating the test set is more balanced across all five classes. The confusion matrix shows 27 correct predictions (all class 4 samples) and 105 incorrect predictions, with all errors being false positives for class 4.

# %%
from sklearn.metrics import cohen_kappa_score

qwk = cohen_kappa_score(y_test, y_pred_dummy, weights='quadratic')
print(f"Quadratic Weighted Kappa: {qwk:.4f}")

# %%
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred_dummy)
print(f"Mean Absolute Error: {mae:.4f}")
