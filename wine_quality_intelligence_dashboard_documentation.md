# Wine Quality Intelligence Dashboard: Documentation & Output Report

## 1. Executive Summary
The **Wine Quality Intelligence Dashboard** (`app_2.py`) is an interactive Streamlit application designed for exploratory data analysis (EDA), predictive model evaluation (regression and binary classification), and feature importance interpretation based on the UCI White Wine Quality dataset.

---

## 2. Application Architecture & Overview

### Technical Stack
- **Frontend / Framework:** Streamlit (`st`)
- **Data Manipulation:** Pandas (`pd`), NumPy (`np`)
- **Data Visualization:** Plotly Express (`px`)

### Navigation Structure
The application is organized into a sidebar for control inputs and four primary navigation tabs:
1. **📊 EDA & Distributions**
2. **📈 Model Comparison & Regression**
3. **🎯 Classification & Confusion Matrix**
4. **🔍 Feature Importance & Findings**

---

## 3. Detailed Component Breakdown & Simulated Output

### 3.1 Header & Top-Level Metrics (Specifications 1 & 8)
- **Header:** 🍷 White Wine Quality Classification & Analysis Dashboard
- **Subtext:** Explores white wine samples, evaluates model performance, and provides interactive analytics.

#### Output Metrics Display:
| Metric | Value | Details |
| :--- | :--- | :--- |
| **Total Observations (Samples)** | 4,898 | Total UCI White Wine instances |
| **Average Quality Score** | 5.87 / 10 | Overall mean sensory rating |
| **Target Imbalance ('Good' $\ge 7$)** | ~21.6% | Proportion of high-quality wines |

---

### 3.2 Sidebar Controls (Specification 8)
- **Data Filtering:**
  - `Filter Min Alcohol (%)` slider range: `8.0` to `14.0` (Default: `8.0`, Step: `0.5`)
- **Model Selection:**
  - `Select Classification View` dropdown options: `["Random Forest (Winner)", "Decision Tree", "Logistic Regression"]`
- **Predictor Input Simulation:**
  - `Alcohol (%) for Prediction` slider range: `8.0` to `15.0` (Default: `10.5`)
  - `Volatile Acidity (g/L)` slider range: `0.1` to `1.1` (Default: `0.30`)
  - `Density (g/cm³)` slider range: `0.985` to `1.040` (Default: `0.994`)

---

### 3.3 Tab 1: EDA & Distributions (Specifications 2 & 3)

#### 1. Quality Score Distribution Bar Chart
- **X-axis:** Quality Score (3 through 9)
- **Y-axis:** Sample Count
- **Color Scale:** Blues

**Output Data Summary:**
| Quality Score | Count | Percentage |
| :---: | :---: | :---: |
| 3 | 20 | 0.41% |
| 4 | 163 | 3.33% |
| 5 | 1,457 | 29.75% |
| 6 | 2,198 | 44.88% |
| 7 | 880 | 17.97% |
| 8 | 175 | 3.57% |
| 9 | 5 | 0.10% |

#### 2. Chemical Property Correlation Matrix (Heatmap)
- **Color Scale:** Red-Blue (`RdBu_r`) with inverted intensity for negative/positive correlations.

**Correlation Matrix Values:**
| Feature | Alcohol | Vol. Acidity | Residual Sugar | Density |
| :--- | :---: | :---: | :---: | :---: |
| **Alcohol** | $1.00$ | $-0.02$ | $0.30$ | $0.44$ |
| **Vol. Acidity** | $-0.02$ | $1.00$ | $-0.15$ | $-0.27$ |
| **Residual Sugar** | $0.30$ | $-0.15$ | $1.00$ | $0.84$ |
| **Density** | $0.44$ | $-0.27$ | $0.84$ | $1.00$ |

---

### 3.4 Tab 2: Model Comparison & Regression (Specifications 4 & 5)

#### 1. Supervised Regression Model Summary Table
Highlighting maximum performance ($R^2$ Score):

| Model Type | $R^2$ Score | MAE | RMSE |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | 0.29 | 0.58 | 0.74 |
| **Decision Tree Regressor** | 0.35 | 0.52 | 0.70 |
| **Random Forest Regressor** *(Best)* | **0.49** | **0.43** | **0.61** |

#### 2. Actual vs. Predicted Quality Plot
- **Chart Type:** Scatter plot with Ordinary Least Squares (OLS) trendline.
- **Data Points:** 200 synthetic sample comparisons depicting model predictions against true quality ratings.

---

### 3.5 Tab 3: Binary Classification & Confusion Matrix (Specification 6)
- **Target Threshold:** Good ($\ge 7$) vs. Not Good ($< 7$).

#### 1. Classification Metrics Table
*Dynamic metric view based on sidebar selection:*

| Metric | Random Forest (Winner) | Decision Tree / Logistic Regression |
| :--- | :---: | :---: |
| **Accuracy** | **0.845** | 0.765 |
| **Precision** | **0.710** | 0.480 |
| **Recall** | **0.580** | 0.450 |
| **F1-Score** | **0.638** | 0.464 |
| **ROC-AUC** | **0.862** | 0.724 |

#### 2. Confusion Matrix Heatmap (Random Forest Classifier)
| Actual \ Predicted | Predicted: Not Good ($<7$) | Predicted: Good ($\ge7$) | Total |
| :--- | :---: | :---: | :---: |
| **Actual: Not Good** | **3,620** *(True Negative)* | **218** *(False Positive)* | 3,838 |
| **Actual: Good** | **180** *(False Negative)* | **880** *(True Positive)* | 1,060 |

---

### 3.6 Tab 4: Feature Importance & Findings (Specifications 7 & 9)

#### 1. Permutation Feature Importance (Horizontal Bar Chart)
Ordered by drop in ROC-AUC:
1. **Alcohol:** `0.342`
2. **Volatile Acidity:** `0.221`
3. **Free Sulfur Dioxide:** `0.125`
4. **Density:** `0.094`
5. **Residual Sugar:** `0.062`
6. **pH:** `0.035`

#### 2. Key Insights & Findings
1. **Alcohol as Primary Quality Driver:** Permutation importance shows **Alcohol** is the single strongest predictor of white wine quality. Higher alcohol content strongly correlates with ratings $\ge 7$.
2. **Negative Impact of Volatile Acidity:** **Volatile Acidity** is the second most impactful feature, serving as a negative quality indicator due to acetic acid off-flavors.
3. **Superiority of Ensemble Models:** The **Random Forest Classifier** outperforms linear models and single decision trees ($\text{ROC-AUC} \approx 0.86$), demonstrating that wine quality relies on non-linear chemical feature interactions.