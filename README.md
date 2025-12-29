# Deep Learning for Intrusion Detection System (IDS)

Network Traffic Classification using Deep Learning on CICIDS2018 Dataset.

## Prerequisites

- **uv**: Fast Python package installer
- **awscli**: AWS Command Line Interface
- **git**: Version control


## Setup

1. **Clone repository**
   ```bash
   git clone <repository-url>
   cd deeplearning-cis2018
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Download dataset**
   ```bash
   aws s3 sync --no-sign-request "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" data/csv/
   ```

4. **Preprocess data**
   ```bash
   uv run data/preprocess_csv_v2.py
   ```

5. **Configure model**
   Edit `config.yaml` and set:
   ```yaml
   model_name: mlp
   ```

6. **Train and test**
   ```bash
   uv run main.py
   ```
   Or open `IDS_Project_Complete.ipynb` and run all cells.

## Expected Results

- Test Accuracy: ~96-97%
- Model saved: `DL-CIS2018.pth`
- Visualizations: Confusion matrix, per-class metrics

## Documentation

- `REPORT.md` - Complete project report
- `IDS_Project_Complete.ipynb` - Step-by-step notebook
- `MODEL_GUIDE.md` - Model architecture guide
