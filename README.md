# Deep Learning Class (VITMMA19) Project Work: LegalTextDecoder

The log file must be uploaded to `log/run.log` to the repository. The logs must be easy to understand and self explanatory.
- [ ] **Logging**:
    - [ ] Log uploaded to `log/run.log`
- [ ] **Docker**:
    - [ ] `Dockerfile` is adapted to your project needs.
    - [ ] Image builds successfully (`docker build -t dl-project .`).
    - [ ] Container runs successfully with data mounted (`docker run ...`).
    - [ ] The container executes the full pipeline (preprocessing, training, evaluation).

## Project Details

### Project Information

- **Selected Topic**: Legal Text Decoder
- **Student Name**: András Jankó
- **Aiming for +1 Mark**: Yes

### Solution Description

#### Problem

Hungarian legal texts, particularly ÁSZF (Általános Szerződési Feltételek / Terms and Conditions), are notoriously difficult to read. This project aims to classify the readability of legal text paragraphs on a 1-5 scale:
- **1** — Nagyon nehezen érthető (Very difficult to understand)
- **2** — Nehezen érthető (Difficult to understand)
- **3** — Többé-kevésbé érthető (Somewhat understandable)
- **4** — Érthető (Understandable)
- **5** — Könnyen érthető (Easy to understand)

#### Data Preprocessing ([01_preprocess.py](src/01_preprocess.py))

The raw aggregated data undergoes several cleaning steps:

1. **Test holdout separation** — Reserves specific annotators (K3I7DL, BCLHKC/otp records) for unbiased evaluation
2. **Duplicate resolution** — When the same text has multiple labels, keeps the annotation with higher lead time (more thoughtful labeling)
3. **Short text filtering** — Removes texts shorter than 40 characters (insufficient content for meaningful classification)
4. **Quality filtering** — Excludes annotators with mean lead time below 10 seconds (rushed, unreliable annotations)

#### Model Architecture

The solution fine-tunes **HuBERT** (`SZTAKI-HLT/hubert-base-cc`), a Hungarian BERT model pre-trained on Common Crawl data. The architecture consists of:

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

### Data Preparation

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

If in the future the SharePoint link is unavailable you can modify the script, by taking out `download_and_extract_data` function and start the execution with a folder structure like this:
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

### Extra Credit Justification

While I may not have invented a revolutionary new architecture or achieved state-of-the-art results, I believe the strength of this submission lies in its completeness and reliability. Everything works. The Docker container builds. The pipeline runs end-to-end. The logs are readable. The code is clean. In the world of machine learning projects, this is rarer than one might hope. I invested considerable time into ensuring that every component — from data preprocessing to evaluation — is well-documented, reproducible, and robust. The kind of thorough, unglamorous work that doesn't make headlines but does make graders' lives easier. In summary: a solid, dependable project that does exactly what it promises, delivered with care and attention to detail. I believe this craftsmanship deserves recognition.

Or perhaps I've simply stared at this code for so long that I've lost all objectivity, and this is, in fact, deeply mediocre. In which case — thank you for reading this far, and I appreciate your patience.

### Docker Instructions

This project is containerized using Docker. Follow the instructions below to build and run the solution.
[Adjust the commands that show how do build your container and run it with log output.]

#### Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker build -t dl-project .
```

#### Run

To run the solution, use the following command. You must mount your local data directory to `/app/data` inside the container.

**To capture the logs for submission (required), redirect the output to a file:**

```bash
docker run -v /absolute/path/to/your/local/data:/app/data dl-project > log/run.log 2>&1
```

*   Replace `/absolute/path/to/your/local/data` with the actual path to your dataset on your host machine that meets the [Data preparation requirements](#data-preparation).
*   The `> log/run.log 2>&1` part ensures that all output (standard output and errors) is saved to `log/run.log`.
*   The container is configured to run every step (data preprocessing, training, evaluation, inference).


### File Structure and Functions

[Update according to the final file structure.]

The repository is structured as follows:

- **`src/`**: Contains the source code for the machine learning pipeline.
    - `01-data-preprocessing.py`: Scripts for loading, cleaning, and preprocessing the raw data.
    - `02-training.py`: The main script for defining the model and executing the training loop.
    - `03-evaluation.py`: Scripts for evaluating the trained model on test data and generating metrics.
    - `04-inference.py`: Script for running the model on new, unseen data to generate predictions.
    - `config.py`: Configuration file containing hyperparameters (e.g., epochs) and paths.
    - `utils.py`: Helper functions and utilities used across different scripts.

- **`notebook/`**: Contains Jupyter notebooks for analysis and experimentation.
    - `01-data-exploration.ipynb`: Notebook for initial exploratory data analysis (EDA) and visualization.
    - `02-label-analysis.ipynb`: Notebook for analyzing the distribution and properties of the target labels.

- **`log/`**: Contains log files.
    - `run.log`: Example log file showing the output of a successful training run.

- **Root Directory**:
    - `Dockerfile`: Configuration file for building the Docker image with the necessary environment and dependencies.
    - `requirements.txt`: List of Python dependencies required for the project.
    - `README.md`: Project documentation and instructions.
