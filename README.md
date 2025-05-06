# ClimateCheck Scientific Claim Verification 
This repository contains code for the ClimateCheck shared task at SDP Workshop (ACL 2025), focused on scientific fact-checking of social media posts about climate change.

##  🧪 Task Overview
- **Subtask I**: Abstract Retrieval 📚 - retrieve top 10 relevant abstracts for climate change claims
- **Subtask II**: Claim Verification ✅❌❓ - classify claim-abstract pairs as 'support', 'refutes', or 'not enough information'

## 🚀 Getting Started
```bash
# Clone repository
git clone [https://github.com/your-username/climatecheck.git](https://github.com/nicolauduran45/climatecheck2025)
cd climatecheck

# Set up environment
python -m venv venv
source venv/bin/activate 
pip install -r requirements.txt

# Download datasets
python src/data/download.py
```
## 📁 Repository Structure
```
climatecheck/
├── configs/                  # Configuration files for experiments
├── data/                     # Dataset storage and processing
│   ├── raw/                  # Original datasets
│   ├── processed/            # Processed datasets
│   └── knowledge_graph/      # Knowledge graph files
├── experiments/              # Experiment scripts
│   ├── retrieval/            # Subtask I experiments
│   └── verification/         # Subtask II experiments
├── models/                   # Model implementations
│   ├── retrieval/            # Retrieval models
│   ├── verification/         # Verification models
│   └── shared/               # Shared model components
├── notebooks/                # Exploratory notebooks
├── outputs/                  # Experiment outputs
│   ├── runs/                 # Training runs
│   ├── predictions/          # Model predictions
│   └── analysis/             # Result analysis
├── references/               # Intersting papers for literature review
│   ├── retrieval/            # About scientific claim retrieval 
│   ├── verification/         # About scientific claim verification
├── src/                      # Source code
│   ├── data/                 # Data processing utilities
│   ├── knowledge/            # Knowledge graph construction
│   ├── evaluation/           # Evaluation metrics and tools
│   └── utils/                # Shared utilities
├── tests/                    # Unit tests
├── .gitignore
├── requirements.txt
└── README.md
```
## 🤝 Contribution Guidelines
1. Branching Strategy
- `main`: Stable code
- `dev`: Development branch
- Feature branches: `feature/your-feature-name`
- Experiment branches: `exp/experiment-name`

2. Experiments
- Store configs in `configs/`
- Save outputs to `outputs/runs/[exp_name]/`
- Include a README in experiment folders

## 🏃‍♂️ Running Experiments
```bash
# Run retrieval experiment
python -m src.run_experiment --config configs/retrieval/dense_retrieval_document_specter.yaml
# Run verification experiment

# Run end-to-end pipeline
```

## 📚 Datasets
- Training/Testing: https://huggingface.co/datasets/rabuahmad/climatecheck
- Publications Corpus: https://huggingface.co/datasets/rabuahmad/climatecheck_publications_corpus

## 🗓️ Important Dates
- Training set release: April 1, 2025
- Test set release: April 15, 2025
- Systems submissions deadline: May 16, 2025
- Paper submission deadline: May 23, 2025
