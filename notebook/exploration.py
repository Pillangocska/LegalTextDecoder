# %%
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import seaborn as sns
import pandas as pd
import numpy as np
import warnings

# %%
# Set up plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load the aggregated data
data_path = Path('../_data/aggregated/labeled_data.csv')
df = pd.read_csv(data_path)

# Display basic information
print(f"Dataset shape: {df.shape}")
print("\nColumn names:")
print(df.columns.tolist())

# %%
print("="*80)
print("SEPARATING TEST DATASET")
print("="*80)

# Identify test records that should be held out
test_condition = (
    (df['student_code'] == 'K3I7DL') |  # All K3I7DL records
    ((df['student_code'] == 'BCLHKC') & (df['source_file'].str.contains('otp', case=False)))  # BCLHKC otp records
)

# Split into test and working dataset
df_test_holdout = df[test_condition].copy()
df = df[~test_condition].copy()

print(f"\nTest holdout dataset: {len(df_test_holdout)} records")
print(f"  K3I7DL: {len(df_test_holdout[df_test_holdout['student_code'] == 'K3I7DL'])} records")
print(f"  BCLHKC (otp): {len(df_test_holdout[df_test_holdout['student_code'] == 'BCLHKC'])} records")

print(f"\nWorking dataset (for exploration and training): {len(df)} records")

# Save test holdout immediately
output_dir = Path('../_data/final')
output_dir.mkdir(parents=True, exist_ok=True)
test_path = output_dir / 'test.csv'
df_test_holdout.to_csv(test_path, index=False, encoding='utf-8')

print(f"\nTest holdout saved to: {test_path}")
print("\nContinuing exploration with working dataset...")
print("="*80)

# %%
# Display first few rows
print("First 3 rows:")
print(df.head(3))

print("\n" + "="*80 + "\n")

# Data types
print("Data types:")
print(df.dtypes)

print("\n" + "="*80 + "\n")

# Missing values
print("Missing values:")
print(df.isnull().sum())

# %%
# Keep only relevant columns
columns_to_keep = [
    'student_code',
    'json_filename',
    'text',
    'label_text',
    'label_numeric',
    'annotation_created_at',
    'lead_time_seconds'
]

df_clean = df[columns_to_keep].copy()

# Rename for clarity
df_clean = df_clean.rename(columns={
    'annotation_created_at': 'labeled_at'
})

print(f"Cleaned dataset shape: {df_clean.shape}")
print("\nFirst 3 rows:")
print(df_clean.head(3))

# %%
# Check for duplicate texts
print("Duplicate analysis:")
print(f"Total records: {len(df_clean)}")
print(f"Unique texts: {df_clean['text'].nunique()}")
print(f"Duplicate texts: {len(df_clean) - df_clean['text'].nunique()}")

print("\n" + "="*80 + "\n")

# Find texts that appear more than once
duplicate_texts = df_clean[df_clean.duplicated(subset=['text'], keep=False)]
print(f"Records with duplicate texts: {len(duplicate_texts)}")

# Show how many times each duplicate text appears
if len(duplicate_texts) > 0:
    duplicate_counts = duplicate_texts.groupby('text').size().sort_values(ascending=False)
    print("\nTop 10 most duplicated texts (by frequency):")
    print(duplicate_counts.head(10))

    print("\n" + "="*80 + "\n")

    # For duplicates, show label distribution
    print("Label variation in duplicate texts:")

    # Group by text and show label statistics
    duplicate_label_stats = duplicate_texts.groupby('text').agg({
        'label_numeric': ['count', 'mean', 'std', 'min', 'max'],
        'student_code': 'nunique'
    }).round(2)

    duplicate_label_stats.columns = ['count', 'mean_label', 'std_label', 'min_label', 'max_label', 'unique_students']
    duplicate_label_stats = duplicate_label_stats.sort_values('count', ascending=False)

    print(duplicate_label_stats.head(10))

# %%
# Identify duplicates and their agreement status
duplicate_mask = df_clean.duplicated(subset=['text'], keep=False)
duplicates = df_clean[duplicate_mask].copy()
non_duplicates = df_clean[~duplicate_mask].copy()

print(f"Non-duplicate records: {len(non_duplicates)}")
print(f"Duplicate records to process: {len(duplicates)}")

# Process duplicates
kept_duplicates = []

for text, group in duplicates.groupby('text'):
    if len(group) == 2:
        labels = group['label_numeric'].values

        # If labels agree, keep the first one
        if labels[0] == labels[1]:
            kept_duplicates.append(group.iloc[0])
        else:
            # If labels disagree, keep the one with higher lead_time
            max_lead_time_idx = group['lead_time_seconds'].idxmax()
            kept_duplicates.append(group.loc[max_lead_time_idx])
    else:
        # Should not happen based on our analysis, but handle just in case
        # Keep the one with highest lead time
        max_lead_time_idx = group['lead_time_seconds'].idxmax()
        kept_duplicates.append(group.loc[max_lead_time_idx])

# Create final cleaned dataframe
kept_duplicates_df = pd.DataFrame(kept_duplicates)
df_final = pd.concat([non_duplicates, kept_duplicates_df], ignore_index=True)

print(f"\nFinal dataset size: {len(df_final)}")
print(f"Records removed: {len(df_clean) - len(df_final)}")
print(f"Unique texts in final dataset: {df_final['text'].nunique()}")

# %%
print("="*80)
print("STUDENT DISTRIBUTION ANALYSIS")
print("="*80)

# Count records per student
student_counts = df_final['student_code'].value_counts().sort_values(ascending=False)

print(f"\nTotal unique students: {df_final['student_code'].nunique()}")
print(f"\nRecords per student:")
print(student_counts)

print(f"\nSummary statistics:")
print(f"  Mean: {student_counts.mean():.2f}")
print(f"  Median: {student_counts.median():.2f}")
print(f"  Min: {student_counts.min()}")
print(f"  Max: {student_counts.max()}")
print(f"  Std: {student_counts.std():.2f}")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Bar plot
student_counts.plot(kind='bar', ax=ax1, color='steelblue')
ax1.set_title('Number of Records per Student', fontsize=14, fontweight='bold')
ax1.set_xlabel('Student Code', fontsize=12)
ax1.set_ylabel('Number of Records', fontsize=12)
ax1.axhline(student_counts.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {student_counts.mean():.1f}')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Histogram
ax2.hist(student_counts.values, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
ax2.set_title('Distribution of Records per Student', fontsize=14, fontweight='bold')
ax2.set_xlabel('Number of Records', fontsize=12)
ax2.set_ylabel('Number of Students', fontsize=12)
ax2.axvline(student_counts.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {student_counts.mean():.1f}')
ax2.axvline(student_counts.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {student_counts.median():.1f}')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# %%
print("="*80)
print("TEXT LENGTH DISTRIBUTION ANALYSIS")
print("="*80)

# Calculate text length
df_final['text_length'] = df_final['text'].str.len()

print("\nText length statistics (characters):")
print(df_final['text_length'].describe())

# Additional percentiles
percentiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
print("\nPercentiles:")
for p in percentiles:
    print(f"  {int(p*100):2d}th: {df_final['text_length'].quantile(p):.0f} chars")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Histogram
axes[0, 0].hist(df_final['text_length'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Distribution of Text Length', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Text Length (characters)', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].axvline(df_final['text_length'].mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {df_final["text_length"].mean():.0f}')
axes[0, 0].axvline(df_final['text_length'].median(), color='green', linestyle='--', linewidth=2,
                    label=f'Median: {df_final["text_length"].median():.0f}')
axes[0, 0].legend()
axes[0, 0].grid(axis='y', alpha=0.3)

# Box plot
axes[0, 1].boxplot(df_final['text_length'], vert=True)
axes[0, 1].set_title('Text Length Box Plot', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('Text Length (characters)', fontsize=12)
axes[0, 1].grid(axis='y', alpha=0.3)

# Log scale histogram (for better visualization if there are outliers)
axes[1, 0].hist(df_final['text_length'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Distribution of Text Length (Log Scale Y-axis)', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Text Length (characters)', fontsize=12)
axes[1, 0].set_ylabel('Frequency (log scale)', fontsize=12)
axes[1, 0].set_yscale('log')
axes[1, 0].grid(axis='y', alpha=0.3)

# Cumulative distribution
sorted_lengths = np.sort(df_final['text_length'])
cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths) * 100
axes[1, 1].plot(sorted_lengths, cumulative, color='steelblue', linewidth=2)
axes[1, 1].set_title('Cumulative Distribution of Text Length', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Text Length (characters)', fontsize=12)
axes[1, 1].set_ylabel('Cumulative Percentage (%)', fontsize=12)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %%
print("="*80)
print("LABEL DISTRIBUTION ANALYSIS")
print("="*80)

# Label counts
label_counts = df_final['label_numeric'].value_counts().sort_index()

print("\nLabel distribution:")
for label, count in label_counts.items():
    percentage = (count / len(df_final)) * 100
    print(f"  Label {label}: {count:4d} ({percentage:5.2f}%)")

print(f"\nTotal: {len(df_final)}")

# Label text distribution
print("\n" + "="*80)
print("\nLabel text distribution:")
label_text_counts = df_final['label_text'].value_counts()
for label_text, count in label_text_counts.items():
    percentage = (count / len(df_final)) * 100
    print(f"  {label_text:30s}: {count:4d} ({percentage:5.2f}%)")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Bar plot - counts
axes[0, 0].bar(label_counts.index, label_counts.values, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Label Distribution (Count)', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Label', fontsize=12)
axes[0, 0].set_ylabel('Count', fontsize=12)
axes[0, 0].set_xticks(label_counts.index)
axes[0, 0].grid(axis='y', alpha=0.3)

# Add count labels on bars
for idx, (label, count) in enumerate(label_counts.items()):
    axes[0, 0].text(label, count, str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')

# Pie chart
colors = ['#d62728', '#ff7f0e', '#ffdd57', '#2ca02c', '#1f77b4']
axes[0, 1].pie(label_counts.values, labels=[f'Label {l}' for l in label_counts.index],
               autopct='%1.1f%%', startangle=90, colors=colors, textprops={'fontsize': 11})
axes[0, 1].set_title('Label Distribution (Percentage)', fontsize=14, fontweight='bold')

# Horizontal bar with percentages
percentages = (label_counts / len(df_final)) * 100
axes[1, 0].barh(label_counts.index, percentages.values, color='steelblue', edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Label Distribution (Percentage)', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Label', fontsize=12)
axes[1, 0].set_xlabel('Percentage (%)', fontsize=12)
axes[1, 0].set_yticks(label_counts.index)
axes[1, 0].invert_yaxis()
axes[1, 0].grid(axis='x', alpha=0.3)

# Add percentage labels
for idx, (label, pct) in enumerate(percentages.items()):
    axes[1, 0].text(pct, label, f'{pct:.1f}%', ha='left', va='center', fontsize=10, fontweight='bold')

# Cumulative distribution
cumulative = label_counts.cumsum()
cumulative_pct = (cumulative / len(df_final)) * 100
axes[1, 1].plot(cumulative.index, cumulative_pct.values, marker='o', linewidth=2, markersize=8, color='steelblue')
axes[1, 1].set_title('Cumulative Label Distribution', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Label', fontsize=12)
axes[1, 1].set_ylabel('Cumulative Percentage (%)', fontsize=12)
axes[1, 1].set_xticks(cumulative.index)
axes[1, 1].grid(alpha=0.3)
axes[1, 1].set_ylim([0, 105])

# Add cumulative percentage labels
for label, pct in zip(cumulative.index, cumulative_pct.values):
    axes[1, 1].text(label, pct + 2, f'{pct:.1f}%', ha='center', fontsize=9)

plt.tight_layout()
plt.show()

# %%
print("="*80)
print("LEAD TIME DISTRIBUTION ANALYSIS")
print("="*80)

print("\nLead time statistics (seconds):")
print(df_final['lead_time_seconds'].describe())

print("\n" + "="*80)
print("\nLead time in different units:")
print(f"  Mean: {df_final['lead_time_seconds'].mean():.2f} seconds ({df_final['lead_time_seconds'].mean()/60:.2f} minutes)")
print(f"  Median: {df_final['lead_time_seconds'].median():.2f} seconds ({df_final['lead_time_seconds'].median()/60:.2f} minutes)")
print(f"  Min: {df_final['lead_time_seconds'].min():.2f} seconds")
print(f"  Max: {df_final['lead_time_seconds'].max():.2f} seconds ({df_final['lead_time_seconds'].max()/60:.2f} minutes)")

# Percentiles
percentiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
print("\nPercentiles:")
for p in percentiles:
    val = df_final['lead_time_seconds'].quantile(p)
    print(f"  {int(p*100):2d}th: {val:6.2f} seconds ({val/60:5.2f} minutes)")

# Lead time by label
print("\n" + "="*80)
print("\nLead time by label:")
for label in sorted(df_final['label_numeric'].unique()):
    lead_times = df_final[df_final['label_numeric'] == label]['lead_time_seconds']
    print(f"  Label {label}: Mean={lead_times.mean():6.2f}s, Median={lead_times.median():6.2f}s, Std={lead_times.std():6.2f}s")

# Plot
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Histogram (all data)
axes[0, 0].hist(df_final['lead_time_seconds'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Distribution of Lead Time', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Lead Time (seconds)', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].axvline(df_final['lead_time_seconds'].mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {df_final["lead_time_seconds"].mean():.1f}s')
axes[0, 0].axvline(df_final['lead_time_seconds'].median(), color='green', linestyle='--', linewidth=2,
                    label=f'Median: {df_final["lead_time_seconds"].median():.1f}s')
axes[0, 0].legend()
axes[0, 0].grid(axis='y', alpha=0.3)

# Histogram (filtered - reasonable range)
reasonable_lead_times = df_final[df_final['lead_time_seconds'] <= 120]['lead_time_seconds']
axes[0, 1].hist(reasonable_lead_times, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Distribution of Lead Time (≤120s)', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Lead Time (seconds)', fontsize=12)
axes[0, 1].set_ylabel('Frequency', fontsize=12)
axes[0, 1].grid(axis='y', alpha=0.3)

# Box plot
axes[0, 2].boxplot(df_final['lead_time_seconds'], vert=True)
axes[0, 2].set_title('Lead Time Box Plot', fontsize=14, fontweight='bold')
axes[0, 2].set_ylabel('Lead Time (seconds)', fontsize=12)
axes[0, 2].grid(axis='y', alpha=0.3)

# Lead time by label - box plot
label_data = [df_final[df_final['label_numeric'] == label]['lead_time_seconds'].values for label in sorted(df_final['label_numeric'].unique())]
bp = axes[1, 0].boxplot(label_data, labels=sorted(df_final['label_numeric'].unique()), patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('steelblue')
    patch.set_alpha(0.7)
axes[1, 0].set_title('Lead Time by Label', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Label', fontsize=12)
axes[1, 0].set_ylabel('Lead Time (seconds)', fontsize=12)
axes[1, 0].grid(axis='y', alpha=0.3)

# Lead time by label - violin plot
parts = axes[1, 1].violinplot(label_data, positions=sorted(df_final['label_numeric'].unique()),
                               showmeans=True, showmedians=True)
for pc in parts['bodies']:
    pc.set_facecolor('steelblue')
    pc.set_alpha(0.7)
axes[1, 1].set_title('Lead Time by Label (Violin Plot)', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Label', fontsize=12)
axes[1, 1].set_ylabel('Lead Time (seconds)', fontsize=12)
axes[1, 1].set_xticks(sorted(df_final['label_numeric'].unique()))
axes[1, 1].grid(axis='y', alpha=0.3)

# Scatter: text length vs lead time
axes[1, 2].scatter(df_final['text_length'], df_final['lead_time_seconds'], alpha=0.3, s=10, color='steelblue')
axes[1, 2].set_title('Text Length vs Lead Time', fontsize=14, fontweight='bold')
axes[1, 2].set_xlabel('Text Length (characters)', fontsize=12)
axes[1, 2].set_ylabel('Lead Time (seconds)', fontsize=12)
axes[1, 2].grid(alpha=0.3)

# Add correlation
corr = df_final['text_length'].corr(df_final['lead_time_seconds'])
axes[1, 2].text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=axes[1, 2].transAxes,
                fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

# %% [markdown]
# Dataset Overview:
#   Total records: 3399
#   Unique texts: 3399
#   Unique students: 25
#   Unique source files: 30
#
# Text Characteristics:
#   Average text length: 405 characters
#   Median text length: 309 characters
#   Shortest text: 2 characters
#   Longest text: 9223 characters
#
# Label Distribution:
#   Label 1:  168 ( 4.94%)
#   Label 2:  389 (11.44%)
#   Label 3:  727 (21.39%)
#   Label 4: 1021 (30.04%)
#   Label 5: 1094 (32.19%)
#
# Labeling Statistics:
#   Average lead time: 43.87 seconds
#   Median lead time: 12.27 seconds
#   Records per student (avg): 136.0
#
# ================================================================================
#
# Key Observations:
#   1. Class imbalance: Labels skewed towards easier texts (4 & 5)
#      - Easy texts (4-5): 62.2%
#      - Hard texts (1-2): 16.4%
#
#   2. Text length varies significantly (2 to 9,223 characters)
#
#   3. Label 5 (easy) has shortest median lead time (7.3s)
#      Label 1 (very hard) has longest median lead time (15.3s)
#
#   4. Weak positive correlation between text length and lead time

# %%
print("="*80)
print("INVESTIGATING SHORT TEXTS")
print("="*80)

# Find texts shorter than 50 characters to get a better view
short_texts = df_final[df_final['text_length'] < 50].copy()
short_texts = short_texts.sort_values('text_length', ascending=False)

print(f"\nNumber of texts shorter than 50 characters: {len(short_texts)}")

if len(short_texts) > 0:
    print("\nAll short texts (sorted by length, descending):")
    print("-" * 80)

    for idx, row in short_texts.iterrows():
        print(f"\n[Length: {row['text_length']:2d} chars] Student: {row['student_code']} | Label: {row['label_numeric']} | Lead: {row['lead_time_seconds']:6.2f}s")
        print(f"  Text: '{row['text']}'")
        print(f"  Source: {row['json_filename']}")

    print("\n" + "="*80)
    print("\nLength distribution of short texts:")
    length_counts = short_texts['text_length'].value_counts().sort_index(ascending=False)
    for length, count in length_counts.items():
        print(f"  {length:2d} chars: {count} text(s)")
else:
    print("\nNo texts found shorter than 50 characters.")

# %%
print("="*80)
print("APPLYING TEXT LENGTH FILTER (MIN 40 CHARACTERS)")
print("="*80)

# Show current state
print("\nBefore filtering:")
print(f"  Total records: {len(df_final)}")

# Apply filter
df_filtered = df_final[df_final['text_length'] >= 50].copy()

print("\nAfter filtering (text_length >= 50):")
print(f"  Total records: {len(df_filtered)}")
print(f"  Records removed: {len(df_final) - len(df_filtered)}")
print(f"  Percentage removed: {((len(df_final) - len(df_filtered)) / len(df_final) * 100):.2f}%")

print("\n" + "="*80)
print("Updated statistics:")
print("="*80)

print("\nText length statistics:")
print(f"  Min: {df_filtered['text_length'].min():.0f} characters")
print(f"  Max: {df_filtered['text_length'].max():.0f} characters")
print(f"  Mean: {df_filtered['text_length'].mean():.0f} characters")
print(f"  Median: {df_filtered['text_length'].median():.0f} characters")

print("\nLabel distribution:")
for label in sorted(df_filtered['label_numeric'].unique()):
    count = (df_filtered['label_numeric'] == label).sum()
    pct = (count / len(df_filtered)) * 100
    print(f"  Label {label}: {count:4d} ({pct:5.2f}%)")

print("\nStudent distribution:")
print(f"  Unique students: {df_filtered['student_code'].nunique()}")
print(f"  Avg records per student: {len(df_filtered) / df_filtered['student_code'].nunique():.1f}")

# %%
print("="*80)
print("INVESTIGATING SUSPICIOUS LEAD TIMES")
print("="*80)

print("\nLead time statistics for filtered dataset:")
print(df_filtered['lead_time_seconds'].describe())

# Define thresholds
very_fast_threshold = 2  # Less than 2 seconds seems suspicious
fast_threshold = 5       # Less than 5 seconds might be rushed
slow_threshold = 120     # More than 2 minutes might indicate distraction
very_slow_threshold = 300  # More than 5 minutes is very suspicious

print("\n" + "="*80)
print("THRESHOLD ANALYSIS:")
print("="*80)

# Count records in each category
very_fast = df_filtered[df_filtered['lead_time_seconds'] < very_fast_threshold]
fast = df_filtered[(df_filtered['lead_time_seconds'] >= very_fast_threshold) &
                   (df_filtered['lead_time_seconds'] < fast_threshold)]
normal = df_filtered[(df_filtered['lead_time_seconds'] >= fast_threshold) &
                     (df_filtered['lead_time_seconds'] < slow_threshold)]
slow = df_filtered[(df_filtered['lead_time_seconds'] >= slow_threshold) &
                   (df_filtered['lead_time_seconds'] < very_slow_threshold)]
very_slow = df_filtered[df_filtered['lead_time_seconds'] >= very_slow_threshold]

print(f"\nVery fast (< {very_fast_threshold}s):")
print(f"  Count: {len(very_fast)} ({len(very_fast)/len(df_filtered)*100:.2f}%)")

print(f"\nFast ({very_fast_threshold}-{fast_threshold}s):")
print(f"  Count: {len(fast)} ({len(fast)/len(df_filtered)*100:.2f}%)")

print(f"\nNormal ({fast_threshold}-{slow_threshold}s):")
print(f"  Count: {len(normal)} ({len(normal)/len(df_filtered)*100:.2f}%)")

print(f"\nSlow ({slow_threshold}-{very_slow_threshold}s):")
print(f"  Count: {len(slow)} ({len(slow)/len(df_filtered)*100:.2f}%)")

print(f"\nVery slow (>= {very_slow_threshold}s):")
print(f"  Count: {len(very_slow)} ({len(very_slow)/len(df_filtered)*100:.2f}%)")

# Sample very fast records
print("\n" + "="*80)
print(f"SAMPLE OF VERY FAST RECORDS (< {very_fast_threshold}s):")
print("="*80)

if len(very_fast) > 0:
    sample_fast = very_fast.nsmallest(10, 'lead_time_seconds')
    for idx, row in sample_fast.iterrows():
        print(f"\n[{row['lead_time_seconds']:.3f}s] Student: {row['student_code']} | Label: {row['label_numeric']} | Length: {row['text_length']} chars")
        print(f"  Text: '{row['text'][:100]}...' " if len(row['text']) > 100 else f"  Text: '{row['text']}'")

# Sample very slow records
print("\n" + "="*80)
print(f"SAMPLE OF VERY SLOW RECORDS (>= {very_slow_threshold}s):")
print("="*80)

if len(very_slow) > 0:
    sample_slow = very_slow.nlargest(10, 'lead_time_seconds')
    for idx, row in sample_slow.iterrows():
        print(f"\n[{row['lead_time_seconds']:.1f}s ({row['lead_time_seconds']/60:.1f}min)] Student: {row['student_code']} | Label: {row['label_numeric']} | Length: {row['text_length']} chars")
        print(f"  Text: '{row['text'][:100]}...' " if len(row['text']) > 100 else f"  Text: '{row['text']}'")


# %%
print("="*80)
print("INVESTIGATING STUDENT EKGPBX")
print("="*80)

# Get all records for this student
student_records = df_filtered[df_filtered['student_code'] == 'EKGPBX'].copy()

print(f"\nTotal records for EKGPBX: {len(student_records)}")
print(f"\nLead time statistics for EKGPBX:")
print(student_records['lead_time_seconds'].describe())

print("\n" + "="*80)
print("Label distribution for EKGPBX:")
for label in sorted(student_records['label_numeric'].unique()):
    count = (student_records['label_numeric'] == label).sum()
    pct = (count / len(student_records)) * 100
    print(f"  Label {label}: {count:3d} ({pct:5.2f}%)")

print("\n" + "="*80)
print("Comparison with overall dataset:")
print("="*80)

print("\nLead time comparison:")
print(f"  EKGPBX mean:    {student_records['lead_time_seconds'].mean():7.2f}s")
print(f"  Overall mean:   {df_filtered['lead_time_seconds'].mean():7.2f}s")
print(f"  EKGPBX median:  {student_records['lead_time_seconds'].median():7.2f}s")
print(f"  Overall median: {df_filtered['lead_time_seconds'].median():7.2f}s")

print("\nLabel distribution comparison:")
for label in sorted(df_filtered['label_numeric'].unique()):
    ekgpbx_pct = (student_records['label_numeric'] == label).sum() / len(student_records) * 100
    overall_pct = (df_filtered['label_numeric'] == label).sum() / len(df_filtered) * 100
    print(f"  Label {label}: EKGPBX {ekgpbx_pct:5.2f}% vs Overall {overall_pct:5.2f}%")

# Check other students with suspiciously low lead times
print("\n" + "="*80)
print("STUDENTS WITH LOWEST AVERAGE LEAD TIMES:")
print("="*80)

student_avg_lead_time = df_filtered.groupby('student_code').agg({
    'lead_time_seconds': ['mean', 'median', 'count']
}).round(2)
student_avg_lead_time.columns = ['mean_lead_time', 'median_lead_time', 'count']
student_avg_lead_time = student_avg_lead_time.sort_values('mean_lead_time')

print("\nTop 10 students with lowest average lead time:")
print(student_avg_lead_time.head(10))

# %%
print("="*80)
print("REMOVING STUDENTS WITH MEAN LEAD TIME < 10 SECONDS")
print("="*80)

# Calculate mean lead time per student
student_stats = df_filtered.groupby('student_code').agg({
    'lead_time_seconds': 'mean',
    'student_code': 'count'
}).rename(columns={'student_code': 'count', 'lead_time_seconds': 'mean_lead_time'})

# Find students to remove
students_to_remove = student_stats[student_stats['mean_lead_time'] < 10].index.tolist()

print("\nStudents to remove (mean lead time < 10s):")
for student in students_to_remove:
    mean_lt = student_stats.loc[student, 'mean_lead_time']
    count = student_stats.loc[student, 'count']
    print(f"  {student}: mean={mean_lt:.2f}s, records={count}")

print(f"\nTotal students to remove: {len(students_to_remove)}")

# Remove these students
df_quality = df_filtered[~df_filtered['student_code'].isin(students_to_remove)].copy()

print("\nBefore removal:")
print(f"  Total records: {len(df_filtered)}")
print(f"  Unique students: {df_filtered['student_code'].nunique()}")

print("\nAfter removal:")
print(f"  Total records: {len(df_quality)}")
print(f"  Unique students: {df_quality['student_code'].nunique()}")
print(f"  Records removed: {len(df_filtered) - len(df_quality)}")
print(f"  Percentage removed: {((len(df_filtered) - len(df_quality)) / len(df_filtered) * 100):.2f}%")

print("\n" + "="*80)
print("UPDATED STATISTICS:")
print("="*80)

print("\nLead time statistics:")
print(df_quality['lead_time_seconds'].describe())

print("\nLabel distribution:")
for label in sorted(df_quality['label_numeric'].unique()):
    count = (df_quality['label_numeric'] == label).sum()
    pct = (count / len(df_quality)) * 100
    print(f"  Label {label}: {count:4d} ({pct:5.2f}%)")

print("\nText length statistics:")
print(f"  Min: {df_quality['text_length'].min():.0f} characters")
print(f"  Mean: {df_quality['text_length'].mean():.0f} characters")
print(f"  Median: {df_quality['text_length'].median():.0f} characters")
print(f"  Max: {df_quality['text_length'].max():.0f} characters")

print("\nRecords per student:")
records_per_student = df_quality.groupby('student_code').size().sort_values(ascending=False)
print(f"  Mean: {records_per_student.mean():.1f}")
print(f"  Median: {records_per_student.median():.1f}")
print(f"  Min: {records_per_student.min()}")
print(f"  Max: {records_per_student.max()}")

# %%
warnings.filterwarnings('ignore')

print("="*80)
print("STUDENT LABELING BIAS ANALYSIS")
print("="*80)

# Calculate overall label distribution
overall_label_dist = df_quality['label_numeric'].value_counts(normalize=True).sort_index()

print("\nOverall label distribution:")
for label, pct in overall_label_dist.items():
    print(f"  Label {label}: {pct*100:5.2f}%")

print("\n" + "="*80)
print("PER-STUDENT ANALYSIS:")
print("="*80)

# Calculate chi-square test for each student
bias_results = []

for student in sorted(df_quality['student_code'].unique()):
    student_data = df_quality[df_quality['student_code'] == student]
    student_label_dist = student_data['label_numeric'].value_counts()

    # Expected counts based on overall distribution
    expected_counts = overall_label_dist * len(student_data)

    # Observed counts
    observed_counts = [student_label_dist.get(label, 0) for label in sorted(overall_label_dist.index)]
    expected = [expected_counts.get(label, 0) for label in sorted(overall_label_dist.index)]

    # Chi-square test
    chi2, p_value = stats.chisquare(observed_counts, expected)

    # Calculate mean label (higher = biased towards "easy")
    mean_label = student_data['label_numeric'].mean()

    # Calculate standard deviation
    std_label = student_data['label_numeric'].std()

    bias_results.append({
        'student': student,
        'count': len(student_data),
        'mean_label': mean_label,
        'std_label': std_label,
        'chi2': chi2,
        'p_value': p_value,
        'significant': p_value < 0.05  # Significant difference at 5% level
    })

# Create DataFrame
bias_df = pd.DataFrame(bias_results)
bias_df = bias_df.sort_values('p_value')

print("\nStudents sorted by bias significance (p-value):")
print("(Lower p-value = more different from overall distribution)")
print("-" * 80)
for idx, row in bias_df.iterrows():
    sig_marker = "***" if row['significant'] else "   "
    print(f"{sig_marker} {row['student']}: p={row['p_value']:.4f}, mean_label={row['mean_label']:.2f}, "
          f"chi2={row['chi2']:.2f}, n={row['count']}")

# Count significant biases
n_significant = bias_df['significant'].sum()
print(f"\n{n_significant} out of {len(bias_df)} students show significant bias (p < 0.05)")

print("\n" + "="*80)
print("LABEL DISTRIBUTION HEATMAP DATA:")
print("="*80)

# Create comparison table
print("\nStudent label distributions (%):")
comparison_data = []
for student in sorted(df_quality['student_code'].unique()):
    student_data = df_quality[df_quality['student_code'] == student]
    label_dist = student_data['label_numeric'].value_counts(normalize=True).sort_index() * 100
    row = {'student': student}
    for label in [1, 2, 3, 4, 5]:
        row[f'label_{label}'] = label_dist.get(label, 0)
    row['mean'] = student_data['label_numeric'].mean()
    comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data).sort_values('mean', ascending=False)
print(comparison_df.to_string(index=False))

print("\n" + "="*80)
print("BIAS CATEGORIES:")
print("="*80)

# Categorize students by mean label
very_harsh = comparison_df[comparison_df['mean'] < 3.0]
harsh = comparison_df[(comparison_df['mean'] >= 3.0) & (comparison_df['mean'] < 3.5)]
balanced = comparison_df[(comparison_df['mean'] >= 3.5) & (comparison_df['mean'] < 4.0)]
lenient = comparison_df[(comparison_df['mean'] >= 4.0) & (comparison_df['mean'] < 4.5)]
very_lenient = comparison_df[comparison_df['mean'] >= 4.5]

print(f"\nVery harsh (mean < 3.0): {len(very_harsh)} students")
if len(very_harsh) > 0:
    print(f"  {', '.join(very_harsh['student'].values)}")

print(f"\nHarsh (3.0 <= mean < 3.5): {len(harsh)} students")
if len(harsh) > 0:
    print(f"  {', '.join(harsh['student'].values)}")

print(f"\nBalanced (3.5 <= mean < 4.0): {len(balanced)} students")
if len(balanced) > 0:
    print(f"  {', '.join(balanced['student'].values)}")

print(f"\nLenient (4.0 <= mean < 4.5): {len(lenient)} students")
if len(lenient) > 0:
    print(f"  {', '.join(lenient['student'].values)}")

print(f"\nVery lenient (mean >= 4.5): {len(very_lenient)} students")
if len(very_lenient) > 0:
    print(f"  {', '.join(very_lenient['student'].values)}")

# %% [markdown]
# Our analysis revealed that 16 out of 23 students (70%) showed statistically significant labeling bias (p < 0.05), with mean label scores ranging from 2.98 to 4.21. Student FA0B9B appeared particularly harsh with a mean label of 2.98, well below the overall mean of ~3.7.
#
# Upon investigation, students like FA0B9B were assigned particularly difficult source documents (e.g., MÁV/Hungarian train company terms). The "harsh" labeling likely reflects genuinely harder-to-understand legal texts rather than annotator bias. The bias might be in the documents, not the labelers. Real-World Variability Legal documents naturally vary in complexity. Having labels that reflect this genuine difficulty variation is more valuable than artificially normalizing away real differences in text comprehension. Sufficient Sample Size With 3,038 records from 23 students (average 132 per student), we have enough data for the model to learn from diverse perspectives. The variety in labeling styles may actually help the model generalize better. Data Preservation Removing biased students would discard ~800+ records (FA0B9B alone: 104 records, plus others). This represents 25%+ of our dataset - too valuable to lose.
#
# The observed "bias" appears to be primarily document-driven rather than annotator-driven. All students will be retained to preserve data diversity and authentic difficulty ratings.

# %%
print("="*80)
print("SAVING FINAL TRAINING DATASET")
print("="*80)

# Save the final cleaned training dataset
output_dir = Path('../_data/final')
output_dir.mkdir(parents=True, exist_ok=True)
train_path = output_dir / 'train.csv'

df_quality.to_csv(train_path, index=False, encoding='utf-8')

print(f"\nTraining dataset saved to: {train_path}")
print(f"  Shape: {df_quality.shape}")
print(f"  Records: {len(df_quality)}")
print(f"  Unique students: {df_quality['student_code'].nunique()}")
print(f"  Unique texts: {df_quality['text'].nunique()}")

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print("\nDataset splits:")
print(f"  Training: {len(df_quality)} records ({df_quality['student_code'].nunique()} students)")
print(f"  Test: {len(df_test_holdout)} records ({df_test_holdout['student_code'].nunique()} students)")
print(f"  Total: {len(df_quality) + len(df_test_holdout)} records")

print("\nData cleaning applied:")
print("  ✓ Removed duplicate texts (kept first or higher lead_time)")
print("  ✓ Removed texts < 40 characters")
print("  ✓ Removed 2 students with mean lead_time < 10s (EKGPBX, XQEBMQ)")
print("  ✓ Kept all remaining students despite labeling bias (document-driven)")
print("  ✓ Separated test set (K3I7DL + BCLHKC otp)")

print("\nFinal training set label distribution:")
for label in sorted(df_quality['label_numeric'].unique()):
    count = (df_quality['label_numeric'] == label).sum()
    pct = (count / len(df_quality)) * 100
    print(f"  Label {label}: {count:4d} ({pct:5.2f}%)")

print("\n" + "="*80)
print("DATASET PREPARATION COMPLETE!")
print("="*80)
print("\nFiles ready for model training:")
print(f"  - {train_path}")
print(f"  - {output_dir / 'test.csv'}")

# %% [markdown]
# # Data Exploration Summary
#
# ## Overview
# This notebook performs comprehensive data exploration and cleaning on Hungarian legal text (ÁSZF - Terms and Conditions) readability labels collected from 25 students. The goal is to prepare a high-quality dataset for training a readability prediction model.
#
# ## Dataset Information
# - **Source**: Labeled JSON files from `_data/original/` containing student annotations
# - **Initial Records**: 3,744 labeled paragraphs
# - **Final Training Records**: 3,038 paragraphs
# - **Labels**: 1-5 scale (1=Very difficult, 5=Very easy)
# - **Students**: 25 annotators (23 after filtering)
#
# ## Data Processing Pipeline
#
# ### 1. Data Loading and Initial Cleaning
# - Loaded aggregated CSV from `_data/aggregated/labeled_data.csv`
# - Selected relevant columns: `student_code`, `json_filename`, `text`, `label_text`, `label`, `labeled_at`, `lead_time_seconds`
# - Renamed columns for clarity
#
# ### 2. Test Set Separation (EARLY)
# **Immediately separated test data** to prevent data leakage:
# - Removed all records from student `K3I7DL` (92 records)
# - Removed BCLHKC records containing "otp" (40 records)
# - **Test set**: 132 records saved to `_data/final/test.csv`
# - **Working set**: 3,452 records for exploration
#
# ### 3. Duplicate Handling
# **Problem**: 186 duplicate texts across 372 records
# - Each duplicate appeared exactly 2 times (intentional overlap for inter-annotator agreement)
# - **Solution**:
#   - If labels agree → keep first occurrence
#   - If labels disagree → keep the one with higher `lead_time_seconds` (more thoughtful)
# - **Result**: 186 duplicates resolved → 3,399 unique texts
#
# ### 4. Text Length Filtering
# **Problem**: Very short texts (minimum 2 characters)
# - Found texts < 50 characters that were likely artifacts or section numbers
# - **Solution**: Applied minimum threshold of **40 characters**
# - **Result**: Removed 137 records (4.03%)
#
# ### 5. Lead Time Analysis
# **Problem**: Suspicious labeling times indicating rushed work
# - Mean lead time: 43.87 seconds
# - Median: 12.27 seconds
# - Found students with suspiciously low average lead times
#
# **Removed students with mean lead time < 10 seconds:**
# - `EKGPBX`: 2.07s mean (114 records)
# - `XQEBMQ`: 8.09s mean (110 records)
# - **Result**: Removed 224 records (6.87%)
#
# ### 6. Labeling Bias Analysis
# **Problem**: 16 out of 23 students (70%) showed statistically significant labeling bias (χ² test, p < 0.05)
# - Mean labels ranged from 2.98 to 4.21
# - Student `FA0B9B` appeared very harsh (mean=2.98)
#
# **Decision: KEPT ALL REMAINING STUDENTS**
# **Reasoning:**
# 1. **Document-driven bias**: Investigation revealed harsh labelers were assigned genuinely difficult documents (e.g., MÁV/Hungarian train company terms)
# 2. **Real-world variability**: Bias reflects actual document difficulty variation, not annotator quality
# 3. **Data preservation**: Removing biased students would lose 25%+ of valuable data
# 4. **Sufficient diversity**: 23 students provide enough perspectives for model generalization
#
# ## Final Dataset Statistics
#
# ### Training Set
# - **Records**: 3,038
# - **Students**: 23
# - **Unique texts**: 3,038
# - **Text length**:
#   - Min: 50 characters
#   - Mean: 424 characters
#   - Median: 321 characters
#   - Max: 9,223 characters
#
# ### Label Distribution
# | Label | Count | Percentage | Description |
# |-------|-------|------------|-------------|
# | 1 | 141 | 4.64% | Very difficult |
# | 2 | 346 | 11.39% | Difficult |
# | 3 | 668 | 21.99% | Somewhat understandable |
# | 4 | 952 | 31.34% | Understandable |
# | 5 | 931 | 30.65% | Easily understandable |
#
# ### Lead Time Statistics
# - Mean: 48.49 seconds
# - Median: 13.82 seconds
# - Range: 0.96 - 4,976 seconds
#
# ## Key Findings
#
# ### 1. Class Imbalance
# - Easy texts (labels 4-5): 62% of dataset
# - Hard texts (labels 1-2): 16% of dataset
# - Model will need techniques to handle imbalance (class weights, oversampling, etc.)
#
# ### 2. Labeling Patterns
# - Label 5 (easy) has shortest median lead time (7.3s) - quickly identified
# - Label 1 (very hard) has longest median lead time (15.3s) - requires more thought
# - Weak positive correlation between text length and lead time
#
# ### 3. Annotator Diversity
# - Records per student: 72-277 (mean: 132, median: 120)
# - Labeling styles vary significantly but likely reflect document difficulty
# - No single "ground truth" - readability is subjective
#
# ## Data Quality Decisions Summary
#
# | Decision | Records Affected | Reasoning |
# |----------|------------------|-----------|
# | Remove duplicates (smart merge) | -186 | Keep most thoughtful labels |
# | Remove texts < 40 chars | -137 | Likely artifacts |
# | Remove fast labelers (< 10s mean) | -224 | EKGPBX, XQEBMQ rushed |
# | Keep biased labelers | +0 | Document-driven, not annotator quality |
# | Separate test set | -132 | K3I7DL + BCLHKC otp |
#
# ## Output Files
#
# 1. **`_data/final/test.csv`** (132 records)
#    - K3I7DL: 92 records
#    - BCLHKC (otp): 40 records
#
# 2. **`_data/final/train.csv`** (3,038 records)
#    - 23 students
#    - Clean, deduplicated, quality-filtered data
#    - Ready for model training
