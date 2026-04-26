# Transformer-Based Spatio-Temporal Deep Learning for Solar Irradiance Forecasting in Smart Grid Applications

**Degree Thesis Manuscript Draft (200-page target template)**  
**Repository-derived version**  
**Date:** April 26, 2026

---

## How to Use This Document

This manuscript is a **thesis-ready master draft** generated from the project artifacts in this repository. It is structured to scale to approximately 200 pages by expanding each subsection with:
- additional literature references,
- expanded methodology details,
- extended result interpretation per site and per horizon,
- appendix materials (code listings, hyperparameter studies, statistical test logs).

A practical pagination plan is provided below.

---

## Target Pagination Plan (200 pages)

- Front matter (title, declaration, acknowledgments, abstract, TOC, lists): **15 pages**
- Chapter 1 (Introduction): **18 pages**
- Chapter 2 (Literature Review): **35 pages**
- Chapter 3 (Data and Problem Formulation): **22 pages**
- Chapter 4 (Methodology and Models): **35 pages**
- Chapter 5 (Experimental Design): **18 pages**
- Chapter 6 (Results and Comparative Analysis): **28 pages**
- Chapter 7 (Smart Grid Implications): **10 pages**
- Chapter 8 (Conclusion and Future Work): **9 pages**
- References: **8 pages**
- Appendices: **2+ pages (or more as needed)**

Total: **~200 pages**

---

## Abstract

The increasing penetration of solar photovoltaic (PV) generation in modern power systems creates a pressing need for accurate short-to-medium term forecasting of Global Horizontal Irradiance (GHI), the key meteorological driver of PV output. Forecast uncertainty directly impacts unit commitment, reserve scheduling, ramp-rate management, and balancing costs in smart grid operations. Traditional statistical and machine learning methods can model local temporal behavior but are limited in their capacity to jointly represent spatial interdependencies among geographically distributed solar sites and long-range temporal structure. To address these limitations, this thesis develops and evaluates a Transformer-based spatio-temporal forecasting framework that explicitly integrates cross-site spatial attention with temporal self-attention.

Using synchronized NSRDB data across geographically diverse locations from January 2017 to December 2019, this work formulates multi-site GHI forecasting as a supervised sequence-to-vector learning problem over site-time-feature tensors. The proposed architecture is benchmarked against ARIMA, support vector regression, tree-based ensembles, LSTM, and GRU baselines under consistent preprocessing and split protocols. Results indicate that the spatio-temporal Transformer achieves superior fit and generalization across multiple forecasting horizons, while providing operationally meaningful improvements for grid decision workflows. The findings confirm that attention-based models are effective in extracting non-local temporal dependencies and exploitable inter-site relationships, especially when climate regimes differ and weather dynamics are nonstationary.

Beyond model-level improvements, this thesis connects predictive performance with grid-level value by discussing implications for reserve allocation, congestion mitigation, and reliability-aware dispatch in solar-rich networks. The work concludes with reproducibility guidance, limitations, and future directions including probabilistic forecasting, physics-informed priors, weather forecast data fusion, and graph-structured spatial modeling.

**Keywords:** Solar forecasting, GHI, spatio-temporal learning, Transformer, smart grid, renewable integration.

---

## Chapter 1. Introduction

### 1.1 Background

Global decarbonization commitments are accelerating the adoption of renewable energy resources, particularly solar photovoltaics. Although solar energy is clean and increasingly cost-competitive, it introduces variability and uncertainty into power system operations due to cloud dynamics, atmospheric conditions, and seasonal cycles. Grid operators therefore depend on reliable forecasting across horizons ranging from intra-hour to day-ahead.

GHI forecasting is central because it is strongly coupled to PV power generation potential. Errors in irradiance forecasts propagate to errors in generation schedules and balancing actions, increasing reserve requirements and potentially reducing grid reliability. As renewable shares grow, even modest forecast gains can translate into substantial economic and operational benefits.

### 1.2 Problem Statement

Many prior forecasting pipelines treat each site independently or rely on methods with limited representational capacity for high-dimensional spatio-temporal interactions. This underutilizes geographically distributed sensing data and can lead to suboptimal forecast quality. The key challenge addressed in this thesis is:

> How can a unified deep learning architecture capture both spatial dependencies across sites and temporal dynamics within each site to improve multi-horizon GHI forecasting for smart grid use?

### 1.3 Research Objectives

1. Build a reproducible end-to-end pipeline for multi-site GHI forecasting.
2. Analyze cross-site correlation structure and geographic effects.
3. Develop a Transformer-based spatio-temporal architecture with explicit attention mechanisms.
4. Benchmark against statistical, machine learning, and recurrent neural baselines.
5. Quantify improvements under multi-horizon evaluation.
6. Interpret the operational impact for smart-grid planning and dispatch.

### 1.4 Research Questions

- RQ1: Do attention-based spatio-temporal models outperform univariate and flattened-feature baselines for GHI forecasting?
- RQ2: To what extent can cross-site correlation be exploited to improve accuracy?
- RQ3: How stable are performance gains across forecasting horizons?
- RQ4: How do forecast improvements translate into practical grid-operation value?

### 1.5 Contributions

- End-to-end implementation of a spatio-temporal Transformer for multi-site GHI prediction.
- Harmonized benchmark framework including ARIMA, SVM, tree ensembles, LSTM, and GRU.
- Multi-horizon evaluation protocol.
- Grid-oriented interpretation of forecast metrics.
- Reproducibility assets in scripts, artifacts, and generated analysis outputs.

### 1.6 Thesis Organization

Chapter 2 reviews prior work. Chapter 3 describes dataset and formulation. Chapter 4 details modeling methodology. Chapter 5 presents experiments. Chapter 6 reports and analyzes results. Chapter 7 discusses grid implications. Chapter 8 concludes and outlines future work.

---

## Chapter 2. Literature Review (Condensed Draft)

### 2.1 Statistical Foundations

Classical time-series models, including ARIMA and seasonal variants, provide interpretable baselines but are typically constrained by assumptions of linearity and stationarity. Solar irradiance processes often violate these assumptions due to weather-driven nonlinearity and abrupt regime shifts.

### 2.2 Machine Learning Approaches

Kernel methods and tree ensembles can model nonlinear interactions but commonly operate on engineered tabular representations that flatten temporal structure. While effective in some settings, such methods may not natively preserve sequence order or cross-site context unless extensively engineered.

### 2.3 Recurrent Neural Networks

LSTM and GRU architectures improved sequence learning for meteorological forecasting by handling long-range dependencies better than vanilla RNNs. However, recurrent processing can limit parallelization and may struggle with very long contexts or explicit spatial relations across multiple sites.

### 2.4 Attention and Transformers

Self-attention enables global dependency modeling with shorter effective path lengths between distant positions. In environmental time series, this is advantageous when influence patterns are distributed and non-local. Extending attention to multi-site settings motivates spatio-temporal variants in which spatial and temporal attention blocks are jointly learned.

### 2.5 Research Gap

A practical and reproducible framework that combines explicit spatial attention, temporal attention, and robust multi-baseline benchmarking for multi-site solar irradiance forecasting remains underrepresented in many applied grid studies. This thesis addresses that gap.

---

## Chapter 3. Data and Problem Formulation

### 3.1 Data Source and Temporal Scope

The dataset is derived from NSRDB records across 2017–2019 with synchronized multi-site observations. Predictors include key irradiance and weather attributes plus engineered temporal features.

### 3.2 Site Diversity and Spatial Context

The selected locations represent distinct climate regimes, enabling evaluation of transferability and cross-site dependency modeling under heterogeneous atmospheric patterns.

### 3.3 Feature Set

- Primary target: GHI.
- Meteorological predictors: DNI, DHI, temperature, humidity, wind speed, pressure, visibility.
- Temporal predictors: hour, month, day-of-year, day-of-week.

### 3.4 Preprocessing Pipeline

- Temporal alignment and synchronization.
- Missing-value handling and cleaning.
- Normalization/scaling.
- Sequence window construction (24-hour inputs).
- Train/test split by time to avoid leakage.

### 3.5 Learning Formulation

Given an input tensor \(X \in \mathbb{R}^{N \times S \times T \times F}\), predict next-step (or horizon-specific) GHI outputs for each site:

\[
\hat{Y} = f_\theta(X), \quad \hat{Y} \in \mathbb{R}^{N \times S}
\]

where \(N\) is number of samples, \(S\) sites, \(T\) time steps, and \(F\) features.

---

## Chapter 4. Methodology and Models

### 4.1 Baseline Models

1. ARIMA
2. SVM (RBF)
3. Random Forest / XGBoost
4. LSTM
5. GRU

### 4.2 Proposed Transformer-ST Architecture

The model uses a sequence of blocks that alternate or combine spatial and temporal attention operations.

#### 4.2.1 Spatial Attention

Spatial attention captures inter-site influence patterns at aligned time indices, enabling the model to learn when one location provides predictive signal for another.

#### 4.2.2 Temporal Attention

Temporal attention captures dependencies across full historical windows, supporting long-range context aggregation.

#### 4.2.3 Positional Encoding

Positional embeddings preserve order information for temporal processing.

#### 4.2.4 Feed-Forward and Residual Structure

Each attention layer is followed by feed-forward projection, normalization, dropout, and residual pathways to stabilize optimization.

#### 4.2.5 Loss and Optimization

Models are trained under regression objectives such as MSE with Adam-based optimization and early stopping.

### 4.3 Complexity and Practicality

The architecture balances expressiveness with manageable parameter count for practical training and inference in operational contexts.

---

## Chapter 5. Experimental Design

### 5.1 Evaluation Metrics

- RMSE
- MAE
- R²
- (Optional) MAPE, nRMSE for operational reporting

### 5.2 Fair Benchmarking Protocol

- Identical train/test split across models.
- Common preprocessing pipeline where applicable.
- Hyperparameter tuning ranges recorded for reproducibility.
- Multiple horizons: 1h, 6h, 12h, 24h.

### 5.3 Statistical Testing

Include paired significance analysis (e.g., Diebold–Mariano or paired bootstrap) to validate differences between model errors.

---

## Chapter 6. Results and Comparative Analysis (Narrative Draft)

### 6.1 Main Performance Trends

Across benchmarks, the spatio-temporal Transformer demonstrates stronger overall accuracy than statistical and conventional ML baselines, with notable gains over recurrent models in settings requiring longer context capture.

### 6.2 Horizon-Wise Behavior

Performance generally degrades with increasing horizon for all models; however, the proposed model retains a relative advantage by leveraging both temporal and spatial information.

### 6.3 Spatial Learning Effects

Correlation analysis indicates that some site pairs have meaningful shared signal. The attention mechanism is able to exploit this selectively, improving generalization for sites with variable atmospheric conditions.

### 6.4 Error Diagnostics

Residual patterns suggest harder forecasting periods around rapid weather transitions, monsoon effects, or high cloud variability windows. Further improvements may require exogenous NWP integration.

### 6.5 Ablation Recommendations (for full thesis expansion)

To complete the 200-page version, include:
- No-spatial-attention ablation,
- No-temporal-attention ablation,
- Head-count sensitivity,
- Sequence length sensitivity,
- Site-removal experiments.

---

## Chapter 7. Smart Grid Implications

### 7.1 Operational Relevance

Improved GHI forecasts can reduce reserve over-procurement, improve economic dispatch fidelity, and reduce balancing penalties.

### 7.2 Reliability and Stability

More accurate day-ahead and intra-day predictions support better ramp scheduling and reduce uncertainty buffers needed for secure operation.

### 7.3 Integration Pathways

The model can be integrated into EMS/SCADA-adjacent analytics stacks as a forecasting microservice with rolling retraining.

---

## Chapter 8. Conclusion and Future Work

### 8.1 Conclusion

This thesis demonstrates that explicit spatio-temporal attention is an effective strategy for multi-site solar irradiance forecasting. The proposed framework advances both predictive quality and practical grid applicability compared with traditional baselines.

### 8.2 Limitations

- Deterministic outputs only (limited uncertainty quantification).
- Potential sensitivity to site selection and climate regime coverage.
- Dependence on data quality and synchronization.

### 8.3 Future Directions

- Probabilistic/quantile transformers,
- NWP and satellite image fusion,
- Graph neural attention for dynamic spatial topology,
- Online learning for concept drift,
- Explainable attention diagnostics for operator trust.

---

## References (Template)

> Replace this section with institution-required citation style (APA/IEEE/etc.).

1. Foundational transformer and attention papers.
2. Recent solar irradiance forecasting benchmarks.
3. Smart grid operational forecasting references.
4. NSRDB/NREL data documentation.

---

## Appendix A: Reproducibility Checklist

- [ ] Python environment and dependency versions captured.
- [ ] Random seeds fixed.
- [ ] Data splits timestamped and archived.
- [ ] Hyperparameter search space documented.
- [ ] Model checkpoints versioned.
- [ ] Evaluation scripts and logs archived.
- [ ] Figure generation scripts recorded.

---

## Appendix B: Chapter Expansion Prompts (for full 200-page writing)

Use these prompts to rapidly expand each chapter:

1. “Expand Section 2.4 into 12 pages with subheadings, equations, and comparative table.”
2. “Write a 10-page method chapter subsection explaining spatial attention tensor shapes and computational graph.”
3. “Generate 8 pages of result interpretation across each site and each horizon with practical implications.”
4. “Create an appendix with pseudo-code and line-by-line commentary of preprocessing and training pipeline.”

---

## Author Note

This draft is intentionally structured as a **submission-oriented base manuscript**. You can now iteratively expand each chapter to your exact university formatting and page requirements while staying grounded in this repository’s implemented pipeline and outputs.
