# Time Series Forecasting for Beijing Air Quality (PM2.5)

A comprehensive machine learning project for predicting hourly PM2.5 air pollution levels in Beijing using LSTM-based deep learning models.

## Project Overview

This project implements various LSTM architectures to forecast air quality (PM2.5 levels) using historical meteorological data, achieving a best validation RMSE of **69.67** through systematic experimentation with different model architectures, feature engineering techniques, and optimization strategies.

## Repository Structure & Navigation

### Core Analysis Notebooks

Navigate through the project evolution:

1. **`air_quality_forecasting_starter_code.ipynb`**

   - **Purpose**: Initial exploration and baseline model
   - **Contains**: Basic data loading, simple LSTM implementation
   - **RMSE Achieved**: ~76-79 (baseline performance)
   - **Start here** for understanding the problem setup

2. **`air_quality_forecasting_1.ipynb`**

   - **Purpose**: Enhanced feature engineering and model improvements
   - **Contains**: Advanced preprocessing, deeper LSTM architectures with BatchNorm
   - **RMSE Achieved**: ~72.37 (significant improvement)
   - **Key innovations**: RobustScaler, extended sequence length (48h), 57 engineered features

3. **`air_quality_forecasting_2.ipynb`**

   - **Purpose**: Optimized model achieving best single-model performance
   - **Contains**: Streamlined architecture with progressive dropout
   - **RMSE Achieved**: **69.67** (best single model)
   - **Key insight**: Sometimes simpler, well-tuned models outperform complex ones

4. **`air_quality_forecasting_3.ipynb`**
   - **Purpose**: Advanced experiments and ensemble methods
   - **Contains**: Bidirectional LSTMs, ensemble techniques
   - **RMSE Achieved**: ~70-71 (ensemble approach)
   - **Focus**: Model diversity and ensemble strategies

### Data Directory (`data/`)

- **`train.csv`**: Training dataset with historical PM2.5 and meteorological features
- **`test.csv`**: Test dataset for final predictions (no PM2.5 labels)

### Documentation

- **`REPORT.md`**: Comprehensive technical report with:
  - Detailed methodology and architecture justifications
  - Complete experiment table with 13+ model variations
  - RMSE formula definition and error pattern analysis
  - RNN challenges (vanishing gradients) and solutions
  - IEEE-formatted citations
  - Performance trends and improvement recommendations

### Submissions Directory (`submissions/`)

Track model performance evolution:

- **`submission_RMSE_69_67_v1.csv`**: Best performing model predictions
- **`FINAL_submission_RMSE_51_75_ensemble_20250921_153557.csv`**: Ensemble submission
- **`enhanced_submission.csv`**, **`improved_submission.csv`**: Progressive improvements
- **`sample_submission.csv`**: Format template

### Additional Files

- **`LICENSE`**: Project licensing information
- **`.git/`**: Version control history

## Recommended Navigation Path

### For Quick Overview:

1. **`REPORT.md`** - Read the executive summary and experiment table
2. **`air_quality_forecasting_2.ipynb`** - See the best performing model

### For Complete Understanding:

1. **`air_quality_forecasting_starter_code.ipynb`** - Understand the baseline
2. **`air_quality_forecasting_1.ipynb`** - See systematic improvements
3. **`air_quality_forecasting_2.ipynb`** - Study the optimized solution
4. **`air_quality_forecasting_3.ipynb`** - Explore advanced techniques
5. **`REPORT.md`** - Comprehensive analysis and conclusions

### For Specific Interests:

- **Feature Engineering**: Focus on notebooks 1-2, preprocessing sections
- **Architecture Design**: Compare model definitions across all notebooks
- **Performance Analysis**: `REPORT.md` experiment table and results sections
- **Reproducibility**: Check random seeds and training configurations in notebooks

## Getting Started

1. **Clone the repository**
2. **Install dependencies**: TensorFlow, pandas, numpy, scikit-learn, matplotlib
3. **Start with**: `air_quality_forecasting_starter_code.ipynb`
4. **Progress through**: Each numbered notebook in sequence
5. **Refer to**: `REPORT.md` for detailed analysis

## Key Achievements

- **Best Single Model RMSE**: 69.67 (ID 13 in experiment table)
- **13+ Systematic Experiments**: Documented with full hyperparameters
- **Multiple Architecture Types**: Single LSTM, Stacked, Deep Regularized, Bidirectional
- **Advanced Techniques**: BatchNorm for vanishing gradients, RobustScaler for outliers
- **Comprehensive Documentation**: Academic-style report with IEEE citations

## Model Performance Summary

| Model Type | Notebook     | RMSE      | Key Innovation                |
| ---------- | ------------ | --------- | ----------------------------- |
| Baseline   | starter_code | ~76-79    | Pipeline establishment        |
| Enhanced   | notebook_1   | 72.37     | Deep architecture + BatchNorm |
| Optimized  | notebook_2   | **69.67** | Simplified + tuned            |
| Ensemble   | notebook_3   | ~70       | Model averaging               |
