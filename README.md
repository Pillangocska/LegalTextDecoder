# Deep Learning Class (VITMMA19) Project Work

## Project Details

### Solution Description

#### Problem

Hungarian legal texts, particularly ÁSZF (Általános Szerződési Feltételek / Terms and Conditions), are notoriously difficult to read. This project aims to classify the readability of legal text paragraphs on a 1-5 scale:
- **1** — Nagyon nehezen érthető (Very difficult to understand)
- **2** — Nehezen érthető (Difficult to understand)
- **3** — Többé-kevésbé érthető (Somewhat understandable)
- **4** — Érthető (Understandable)
- **5** — Könnyen érthető (Easy to understand)

#### Data Preparation

The data preparation is **fully automated** via [00_aggregate_jsons.py](src/00_aggregate_jsons.py):

1. **Downloads** a ZIP file from SharePoint (URL configured in `config.yaml`)
2. **Extracts** to `data/original/` (replaces existing data to ensure freshness)
3. **Processes** JSON files from multiple annotator folders containing labeled ÁSZF paragraphs rated on a 1-5 readability scale
4. **Outputs** `data/aggregated/labeled_data.csv` with all labeled data and metadata

No manual data preparation is required. Simply run:

```bash
python -m src.00_aggregate_jsons  # Downloads and aggregates data
```

Or run the full pipeline via Docker (see [Docker Instructions](#docker-instructions)). This is the ideal method!

If in the future the SharePoint link is unavailable and you have some annotated data of your own, you can modify the script, by taking out `download_and_extract_data` function and start the execution with a folder structure like this:
```bash
📦data
 ┣ 📂original
 ┃ ┣ 📂<NEPTUN/STUDENT_CODE>
 ┃ ┃ ┣ 📜some_company_aszf.txt
 ┃ ┃ ┣ 📜labeled1.json
 ┃ ┃ ┣ 📜labeled2.json
 ┃ ┃ ┗ 📜labeledN.json
 ┃ ┣ 📂<NEPTUN/STUDENT_CODE>
 ┃ ┃ ┣ 📜labeled1.json
 ┃ ┃ ┣ 📜some_company_aszf.txt
 ┃ ┃ ┣ ...
```

#### Data Preprocessing ([01_preprocess.py](src/01_preprocess.py))

The raw aggregated data undergoes several cleaning steps:

1. **Test holdout separation** — Reserves specific annotators (K3I7DL, BCLHKC/otp records) for unbiased evaluation
2. **Duplicate resolution** — When the same text has multiple labels, keeps the annotation with higher lead time (more thoughtful labeling)
3. **Short text filtering** — Removes texts shorter than 40 characters (insufficient content for meaningful classification)
4. **Quality filtering** — Excludes annotators with mean lead time below 10 seconds (rushed, unreliable annotations)

#### Model Architecture

The solution fine-tunes **HuBERT** (`SZTAKI-HLT/hubert-base-cc`), a Hungarian BERT model pre-trained on Common Crawl data and a snapshot of the hungarian Wikipedia. The architecture consists of:

- **Base model**: HuBERT encoder (12 transformer layers, 768 hidden dimensions)
- **Classification head**: Linear layer mapping to 5 readability classes
- **Total parameters**: ~111M (all trainable during fine-tuning)

#### Training Methodology

- **Tokenization**: Maximum sequence length of 512 tokens with padding and truncation
- **Class weighting**: Balanced weights computed from training distribution to handle class imbalance
- **Optimizer**: AdamW with linear learning rate schedule and warmup
- **Loss function**: Cross-entropy with class weights
- **Validation**: Stratified train/validation split (configurable ratio)
- **Checkpointing**: Best model saved based on validation accuracy

#### Evaluation

The trained model is evaluated using:
- **Classification metrics**: Per-class precision, recall, F1-score
- **Ordinal metrics**: Exact accuracy, within-1 accuracy, within-2 accuracy (accounting for the ordinal nature of readability scores)
- **Visualizations**: Confusion matrices for validation and test sets

#### Results

Training was performed for **5 epochs**. The model shows clear learning with validation accuracy stabilizing around epoch 3-4.

![Training and Validation Loss/Accuracy](media/trainloss.png)

**Final metrics:**

| Metric | Validation | Test |
|--------|------------|------|
| Exact Accuracy | 46% | 38% |
| Within-1 Accuracy | 88% | 78% |

The within-1 accuracy metric is particularly relevant for ordinal classification — a prediction of "4" when the true label is "5" is much better than predicting "1". The model achieves ~78-88% within-1 accuracy, meaning predictions are almost always within one readability level of the ground truth.

**Confusion Matrices:**

| Validation | Test |
|------------|------|
| ![Validation Confusion Matrix](media/confmtx_validation.png) | ![Test Confusion Matrix](media/confmtx_test.png) |

The confusion matrices show that the model tends to predict upper-middle classes (4-5) more frequently, which is expected given class imbalance and the inherent subjectivity of readability assessment.

### Docker Instructions

This project is containerized using Docker. Follow the instructions below to build and run the solution.

#### Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker build -t dl-project-nhvu6n .
```

#### Run

To run the solution, use the following command. You must mount your local data directory to `/app/data` inside the container. Also make sure you create a log/ folder from where you are running the script or just run it simply with `> run.log 2>&1` at the end.

```bash
docker run --rm --gpus all -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output dl-project-nhvu6n > log/run.log 2>&1
```

*   Replace `$(pwd)/data` with the actual path to your dataset on your host machine that meets the [Data preparation requirements](#data-preparation). By default that can be an empty folder since we download everything anyway, and docker will create this directory for us.
*   The `> log/run.log 2>&1` part ensures that all output (standard output and errors) is saved to `log/run.log`.
*   The container is configured to run every step (downloading/aggregation, data preprocessing, training, evaluation, inference).
*   After the pipeline is complete and the `/app/output` folder was mounted to a local folder the `best_model.pt` model file, the `training_history.png`, and the confusion matrix diagrams will be available for future use.

If GPU is not available we can run it in CPU-only mode, but in this case it's highly recommended to set the `num_epochs` parameter to 1 in the `config.yaml`, since it will run for around 30 minutes even with one epoch!

### File Structure

```
📦 LegalTextDecoder
├── 📂 src/                          # Main pipeline scripts
│   ├── 00_aggregate_jsons.py        # Downloads data from SharePoint & aggregates JSONs
│   ├── 01_preprocess.py             # Data cleaning & train/test split creation
│   ├── 02_train.py                  # HuBERT fine-tuning training loop
│   ├── 03_evaluation.py             # Model evaluation on validation/test sets
│   ├── 04_inference.py              # Inference on new texts
│   └── 📂 util/
│       ├── config_manager.py        # YAML configuration loader
│       └── logger.py                # Logging utilities
│
├── 📂 notebook/                     # Jupyter notebooks for experimentation
│   ├── 01_data_exploration.ipynb    # Initial EDA and visualization
│   ├── 02_baseline.ipynb            # Baseline model experiments
│   ├── 03_preprocess.ipynb          # Preprocessing experiments
│   ├── 04_training_huBERT.ipynb     # HuBERT training experiments
│   ├── 04_training_*.ipynb          # Other model experiments (XLM-RoBERTa, mBERT, classical NLP etc.)
│   └── 04_training_eval_inf.ipynb   # Combined training/eval/inference notebook
│
├── 📂 media/                        # Result visualizations
│   ├── trainloss.png                # Training/validation loss & accuracy curves
│   ├── confmtx_validation.png       # Validation set confusion matrix
│   └── confmtx_test.png             # Test set confusion matrix
│
├── 📂 log/                          # Log files
│   └── run.log                      # Example pipeline execution log
│
├── Dockerfile                       # Docker container configuration
├── run.sh                           # Pipeline entry point script
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # Project metadata and build configuration
├── uv.lock                          # Locked dependencies (uv package manager)
├── config.yaml                      # Hyperparameters and paths configuration
└── README.md                        # Project documentation and description
```

### Testing the Final Solution

The solution was tested on a fresh cloud GPU instance to verify reproducibility.

#### Cloud Instance Setup

A Lambda Cloud instance was provisioned with the following configuration:

| Property | Value |
|----------|-------|
| Type | gpu_1x_a100_sxm4 |
| Region | us-east-1 |

#### SSH Connection

```bash
ssh -i "path\to\pem\key\key.pem" ubuntu@<IP_Address>
```

#### Execution Steps

1. **Clone the repository:**
```bash
git clone https://github.com/Pillangocska/LegalTextDecoder.git
cd LegalTextDecoder/
```

2. **Build the Docker image** (requires root privileges):
```bash
sudo su
docker build -t dl-project-nhvu6n .
```

3. **Run the full pipeline:**
```bash
docker run --rm --gpus all -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output dl-project-nhvu6n > log/run.log 2>&1
```

The pipeline completed successfully, demonstrating that the solution is fully reproducible on a fresh environment with GPU support.
