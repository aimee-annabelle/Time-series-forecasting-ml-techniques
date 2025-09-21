# Air Quality Time Series Forecasting Report

Repository: https://github.com/aimee-annabelle/Time-series-forecasting-ml-techniques

Easy to read Report: [Notion page of full report](https://surf-passive-c5d.notion.site/Air-pollution-time-series-forecasting-formative-assignment-275042f37c5b809cbb94e78f31151ada?source=copy_link)

## 1. Introduction

The task is to forecast future hourly PM2.5 air pollution levels in Beijing using historical meteorological, calendar, and categorical features. The target variable (`pm2.5`) exhibits high variance and occasional extreme spikes, making robust modeling and feature scaling important.

Approach so far:

- Clean and structure the time series (reconstruct a proper DateTime index from year/month/day/hour fields).
- Handle missing target values by removing rows with missing `pm2.5` (retaining 94% of data).
- Engineer temporal and categorical features (hour, day, month, year, season, weekend indicator, wind direction encoding, etc.).
- Scale features (initially StandardScaler, later RobustScaler to reduce outlier influence).
- Frame the problem as supervised learning with sliding windows (sequence length 24 → later 48).
- Iteratively develop LSTM-based architectures, increasing depth, normalization, and regularization.

Goal was: Achieve progressively lower validation RMSE (both on scaled and original target) while preventing overfitting and improving generalization to the test set used for competition submissions.

## 2. Data Exploration & Preprocessing

### 2.1 Dataset Analysis

The Beijing PM2.5 dataset contains hourly air quality measurements with meteorological features. Statistical analysis revealed:

- Train shape (after removing missing target rows): ≈ 30k rows; Test: 13,148 rows.
- Target distribution: heavily right-skewed with occasional extreme spikes (values > 500 μg/m³, while typical values range 20-150 μg/m³). This distribution motivated RobustScaler adoption over StandardScaler.
- Time series plots showed clear seasonal patterns (higher PM2.5 in winter months) and weekly cyclicity (weekday vs weekend differences).
- Correlation analysis between meteorological features (temperature, pressure, wind speed/direction) and PM2.5 revealed moderate relationships, justifying their inclusion.
- Missing value analysis: 6% of target values were NaN, distributed somewhat randomly across time periods rather than in continuous blocks.

### 2.2 Preprocessing Pipeline

- **Missing data handling**: Rows with NaN PM2.5 values were removed rather than imputed to maintain data integrity for supervised learning. This approach prevents the model from learning spurious patterns from imputed values.
- **Feature engineering rationale**:
    - Temporal features (hour, day, month, season) capture cyclical patterns essential for air quality prediction
    - Wind direction encoding provides categorical meteorological context affecting pollutant dispersion
    - Weekend indicators capture anthropogenic emission differences
- **Scaling strategies**:
    - Baseline: StandardScaler on features and target
    - Improved: RobustScaler to mitigate effect of extreme outliers (>500 μg/m³ spikes) which would skew StandardScaler statistics
- **Sequence framing**: `create_sequences` utility produces (X, y) sliding windows. Window length progression (24→48 hours) was motivated by autocorrelation analysis showing significant PM2.5 persistence up to 48 hours, enabling the model to capture both diurnal cycles and multi-day pollution episodes.

## 3. Model Design

Three main architecture tiers explored:

### 3.1 Baseline Minimal LSTM

```
LSTM(32, activation='tanh') → Dropout(0.2) → Dense(1, activation='linear')
```

**Architecture Justification**:

- Single LSTM layer (32 units) with tanh activation provides sufficient capacity for basic temporal patterns while remaining computationally efficient
- Moderate dropout (0.2) prevents overfitting without excessive regularization
- Linear output activation for regression task
- Adam optimizer (lr=0.001) chosen for adaptive learning rate and momentum properties suitable for RNN training
- Architecture serves as performance floor and pipeline validation

### 3.2 Stacked LSTM

```
LSTM(50, return_sequences=True) → Dropout(0.2)
→ LSTM(50) → Dropout(0.2)
→ Dense(25, activation='relu') → Dense(1)
```

Rationale: Adds representational capacity; second LSTM layer improves modeling of medium-range temporal patterns.

### 3.3 Improved Deep Regularized LSTM (Current Best)

```
LSTM(96, return_sequences=True) → BatchNorm → Dropout(0.3)
→ LSTM(64, return_sequences=True) → BatchNorm → Dropout(0.3)
→ LSTM(32) → BatchNorm → Dropout(0.3)
→ Dense(48, 'relu') → BatchNorm
→ Dense(24, 'relu')
→ Dense(8, 'relu')
→ Dense(1)
```

**Enhancements & RNN Challenge Solutions**:

- **Deeper hierarchy**: Multi-scale temporal abstraction with decreasing layer sizes (96→64→32) to capture both fine and coarse temporal patterns
- **Batch Normalization**: Applied after each LSTM layer to address **vanishing gradient problem** - a key RNN challenge where gradients diminish through time steps, hampering learning of long-term dependencies. BatchNorm normalizes internal covariate shift, stabilizing gradient flow.
- **Gradient clipping**: Implicitly handled by Adam optimizer to prevent **exploding gradients**, another common RNN issue
- **Dropout progression**: Higher dropout (0.3) in deeper network prevents overfitting while maintaining learning capacity
- **Extended context**: 48-hour sequences + 57 engineered features capture longer temporal dependencies and richer environmental context
- **Learning rate scheduling**: ReduceLROnPlateau enables fine-tuning after initial convergence

**Architecture Diagram Concept**:

```
Input(48, 57) → LSTM(96,rs) → BN → DO(0.3) →
LSTM(64,rs) → BN → DO(0.3) → LSTM(32) → BN → DO(0.3) →
Dense(48,relu) → BN → Dense(24,relu) → Dense(8,relu) → Dense(1)

```

Result: Reduced validation RMSE from 76 to 69 (7% improvement).

## 4. Experiment Table

| ID | Architecture (summary) | lstm_units | dropout_rate | learning_rate | batch_size | optimizer | early_stopping_patience | RMSE_original |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | LSTM(24) -> Dropout(0.3) -> Dense(1) | [24] | 0.3 | 0.001 | 32 | Adam | 15 | 77.62 |
| 2 | LSTM(32) -> Dropout(0.35) -> Dense(1) | [32] | 0.35 | 0.001 | 32 | Adam | 15 | 78.42 |
| 3 | LSTM(32) -> Dropout(0.3) -> Dense(1) | [32] | 0.3 | 0.001 | 48 | Adam | 15 | 79.09 |
| 4 | LSTM(24) -> Dropout(0.3) -> LSTM(24) -> Dropout(0.3) -> Dense(1) | [24,24] | 0.3 | 0.001 | 32 | Adam | 15 | 79.21 |
| 5 | LSTM(32) -> Dropout(0.3) -> LSTM(32) -> Dropout(0.3) -> Dense(1) | [32,32] | 0.3 | 0.001 | 32 | Adam | 15 | 79.40 |
| 6 | LSTM(32) -> Dropout(0.2) -> Dense(1) | [32] | 0.2 | 0.001 | 32 | Adam | 15 | 76.09 |
| 7 | LSTM(50, rs) -> Dropout(0.2) -> LSTM(50) -> Dropout -> Dense(25)->Dense(1) | [50,50] | 0.2 | 0.001 | 32 | Adam | 15 | 75.85 |
| 8 | LSTM(96, rs)->BN->DO(0.3)->LSTM(64, rs)->BN->DO(0.3)->LSTM(32)->BN->DO(0.3)->Dense(48)->BN->Dense(24)->Dense(8)->Dense(1) | [96,64,32] | 0.3 | 0.001 | 32 | Adam | 25 | 72.37 |
| 9 | Bidirectional(LSTM(192)) -> Dropout(0.15) -> LSTM(96) -> Dropout(0.15) -> LSTM(48) -> Dense(128) -> Dropout(0.2) -> Dense(1) | [192,96,48] | 0.15-0.2 | 0.0005 | 32 | Adam | 10 | 73 |
| 10 | Bidirectional(LSTM(128, rs)) -> Bidirectional(LSTM(64, rs)) -> Bidirectional(LSTM(32)) -> Dense(64) -> Dropout(0.3) -> Dense(32) -> Dropout(0.2) -> Dense(16) -> Dense(1) | [128,64,32] | 0.2-0.3 | 0.0008 | 32 | Adam | 15 | 71 |
| 11 | LSTM(128) -> LSTM(64) -> LSTM(32) -> Dense(64) -> Dropout(0.3) -> Dense(32) -> Dropout(0.2) -> Dense(1) | [128,64,32] | 0.2-0.3 | 0.0008 | 32 | Adam | 15 | 72 |
| 12 | Ensemble (weighted avg of 10 & 11) | various | various | various | 32 | Adam | 15 | 70 |
| 13 | LSTM(64)->BN->DO(0.25)->Dense(32,relu)->BN->DO(0.15)->Dense(16,relu)->DO(0.1)->Dense(1) | [64] | 0.1-0.25 | 0.002 | 32 | Adam(clipnorm=1.0) | 20 | 69.67 |

## 5. Results Analysis

### 5.1 RMSE Definition and Formula

Root Mean Square Error (RMSE) quantifies prediction accuracy by penalizing larger errors more than smaller ones, making it suitable for air quality forecasting where extreme mispredictions have higher consequences.

**RMSE Formula**:
**RMSE = √[Σ(Actualᵢ - Predictedᵢ)² / n] (RMSE = square root of (sum of squared differences between actual and predicted values, divided by number of predictions)**

Where:

- $n$ = number of predictions
- $y_i$ = actual PM2.5 value
- $\hat{y}_i$ = predicted PM2.5 value

### 5.2 Experimental Results and Performance Trends

Progression from single-layer LSTM (77-79 RMSE range) to deeper architectures revealed key insights:

**Performance Trajectory**:

- Early small models (24-32 units): RMSE 77.62-79.40 - **underfitting** due to insufficient capacity
- Baseline optimization (32 units, 0.2 dropout): RMSE 76.09 - balanced capacity/regularization
- Stacked approach (50+50 units): RMSE 75.85 - marginal improvement
- Deep regularized model: RMSE 72.37 - significant 5.3% improvement from combined enhancements

### 5.2 Experimental Results and Trends

Progression from single-layer LSTM (77-79 RMSE range) to deeper architectures revealed key insights:

**Performance Trajectory**:

- Early small models (24-32 units): RMSE 77.62-79.40 - **underfitting** due to insufficient capacity
- Baseline optimization (32 units, 0.2 dropout): RMSE 76.09 - balanced capacity/regularization
- Stacked approach (50+50 units): RMSE 75.85 - marginal improvement, suggesting architecture limitations
- Deep regularized model: RMSE 72.37 - significant gain from combined improvements

What helped most:

- Longer sequence length: gave the model context for slower-moving pollution patterns.
- Feature expansion: calendar + categorical encodings added signal that a shallow model alone couldn’t tease out.
- Depth + BatchNorm: stabilized training so the network could benefit from additional capacity without immediately overfitting.
- Moderate dropout (0.3): enough regularization to generalize without collapsing learning.

### 5.3 Error Pattern Analysis

**Overfitting vs Underfitting Assessment**:

- Early models (IDs 0-0e): **Underfitting** - high training and validation RMSE, insufficient model capacity
- Baseline model (ID 6): Well-balanced - reasonable gap between training/validation performance
- Deep model (ID 8): **Slight overfitting tendency** controlled by dropout and BatchNorm - validation RMSE stabilized

**Prediction vs Actual Analysis** (qualitative observations from validation):

- **Systematic underprediction** during extreme pollution episodes (>300 μg/m³)
- **Oversmoothing** of rapid day-to-day fluctuations, suggesting model favors temporal averaging
- **Strong performance** on moderate pollution levels (50-200 μg/m³ range)
- **Seasonal bias**: Slight underestimation during winter peaks

**Model Limitations**:

- Sharp upward spikes (sudden pollution events) are consistently under-predicted
- Rapid transitions smoothed due to LSTM's tendency toward temporal averaging
- Limited capacity to model extreme value distributions despite RobustScaler

**Future Improvements**: Attention mechanisms, temporal convolutions, or explicit volatility features could address these patterns.

## 6. Conclusion

### 6.1 Problem Summary

This project addressed hourly PM2.5 air pollution forecasting in Beijing using LSTM-based deep learning models. The challenge involved handling highly variable, skewed target distributions with extreme outliers while capturing complex temporal dependencies.

### 6.2 Approach and Key Findings

**Methodology**: Progressive architectural development from simple baseline (76.09 RMSE) to deep regularized LSTM (72.37 RMSE), incorporating BatchNorm to address vanishing gradients, extended temporal context (48 hours), and comprehensive feature engineering.

**Successes**:

- 5.3% RMSE improvement through systematic architecture enhancement
- Effective handling of RNN training challenges (vanishing/exploding gradients)
- Robust preprocessing pipeline handling missing data and extreme outliers
- Feature engineering capturing temporal and meteorological patterns

**Challenges**:

- Systematic underprediction of extreme pollution events
- Temporal oversmoothing limiting rapid transition accuracy
- Model conservatism in high-variance conditions

### 6.3 Improvement Recommendations

1. **Alternative architectures**: Implement attention mechanisms for spike detection and temporal convolutions for rapid transitions
2. **Feature enhancement**: Add rolling volatility measures and recent extrema features
3. **Ensemble methods**: Combine conservative LSTM with aggressive predictors for extreme events
4. **Loss function modification**: Implement weighted MSE penalizing extreme value mispredictions
5. **Regularization tuning**: Fine-tune dropout rates and BatchNorm placement through systematic grid search

## 7. Citations

1. Hochreiter, S. & Schmidhuber, J. (1997). "Long Short-Term Memory." Neural Computation 9(8): 1735–1780.
2. Ioffe, S. & Szegedy, C. (2015). "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift." ICML.
3. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). "Dropout: A Simple Way to Prevent Neural Networks from Overfitting." JMLR 15: 1929–1958.
4. Kingma, D. P. & Ba, J. (2015). "Adam: A Method for Stochastic Optimization." ICLR.
5. Pedregosa, F. et al. (2011). "Scikit-learn: Machine Learning in Python." Journal of Machine Learning Research 12: 2825–2830. (RobustScaler implementation)
6. Keras Team. "ReduceLROnPlateau Callback Documentation." https://keras.io/api/callbacks/reduce_lr_on_plateau/
7. KDD Cup / Beijing PM2.5 Data Set (original hourly air quality measurements). Frequently cited public dataset used for air pollution forecasting tasks.
8. Chollet, F. et al. "Keras: Deep Learning for humans." [https://keras.io](https://keras.io/)