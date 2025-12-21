# DMMF Replication Package

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18012186.svg)](https://doi.org/10.5281/zenodo.18012186)

Replication package for the paper:

> **From Commits to Cognition: A Mixed-Methods Framework for Inferring Developer Mental Models from Repository Artifacts**

Published: Zenodo, December 2025 | DOI: [10.5281/zenodo.18012186](https://doi.org/10.5281/zenodo.18012186)

## Overview

This repository contains data, scripts, and figures for replicating the Developer Mental Model Framework (DMMF) study. The framework infers cognitive patterns from software repository artifacts using:

- Mining Software Repositories (MSR)
- BERT-based sentiment analysis
- Temporal clustering analysis
- Reflexive Thematic Analysis

## Repository Structure

```
├── data/
│   ├── commits.csv                    # 2,799 commits with metadata
│   ├── repos.json                     # 40 repository metadata
│   ├── summary_stats.json             # Aggregate statistics
│   └── enhanced_analysis_results.json # BERT + temporal analysis
├── figures/
│   ├── commit_heatmap.png             # Activity by day × hour
│   ├── commits_over_time.png          # Monthly commit frequency
│   ├── commit_types.png               # Conventional commit distribution
│   ├── bert_sentiment.png             # Sentiment analysis results
│   └── refactor_patterns.png          # Temporal clustering
└── scripts/
    ├── msr_analysis.py                # Data extraction
    └── enhanced_analysis.py           # BERT + temporal analysis
```

## Quick Start

### Requirements

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas matplotlib seaborn requests scipy transformers torch
```

### Run Analysis

```bash
# Basic MSR analysis
python scripts/msr_analysis.py

# Enhanced analysis (BERT + temporal)
python scripts/enhanced_analysis.py
```

### Adapt for Your Own Repositories

Edit `scripts/msr_analysis.py`:

```python
GITHUB_USER = "your-username"  # Change this
```

## Key Findings

| Metric | Value |
|--------|-------|
| Total commits analyzed | 2,799 |
| Repositories | 40 |
| Date range | Nov 2022 – Dec 2025 |
| BERT Positive sentiment | 18.0% |
| Reflection-in-action (clustered) | 83.4% |
| Reflection-on-action (isolated) | 16.6% |

## DMMF Dimensions

| Dimension | Artifacts Used | Example Finding |
|-----------|----------------|-----------------|
| Cognitive | Directory structure, code metrics | Hierarchical modularity (4.2 levels avg) |
| Affective | Commit sentiment | feat=50% positive, fix=5% positive |
| Conative | Project themes, documentation | Technology as civic tool |
| Reflective | Refactor patterns | 83% clustered refactors |

## Model Information

Sentiment analysis uses `distilbert-base-uncased-finetuned-sst-2-english` from HuggingFace (pre-trained on SST-2). This is a standard model, not custom fine-tuned.

## Citation

```bibtex
@article{silva2025commits,
  title={From Commits to Cognition: A Mixed-Methods Framework for
         Inferring Developer Mental Models from Repository Artifacts},
  author={Silva, Anderson Henrique da},
  journal={arXiv preprint},
  year={2025}
}
```

## License

- Code and data: MIT License
- Paper: CC-BY 4.0
