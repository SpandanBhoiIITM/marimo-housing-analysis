# 🏠 Interactive Housing Market Data Analysis Notebook

## Overview
This project is an **interactive data analysis notebook** that demonstrates relationships between housing prices, property size, age, and location using a **synthetic housing dataset**. Users can dynamically explore the dataset through interactive sliders and dropdowns while visualizations and statistics update in real time.

The notebook leverages **Marimo**, a reactive notebook framework, to create a smooth, interactive experience suitable for data exploration, teaching, and demonstrations.

---

## Features

### 🔄 Interactive Controls
- **Minimum Square Feet Slider** – Filter properties by minimum size.
- **Maximum House Age Slider** – Filter properties by maximum age.
- **Location Dropdown** – Focus analysis on a specific location or all areas.

### 📊 Dynamic Visualizations
- Price vs Square Feet scatter plot with trend line.
- Price distribution by location.
- House age vs price scatter plot.
- Predictive modeling with linear regression and R² metric.

### 📝 Analysis Outputs
- Dynamic markdown summary showing key statistics and correlations.
- Market recommendations based on filtered dataset.
- Interactive data tables:
  - Sample data preview
  - Summary statistics
  - Correlation matrix

### 📈 Modeling
- Linear Regression predicting housing prices based on:
  - Square feet
  - Number of bedrooms
  - House age

### 🏠 Domain-Specific Insights
- Correlation analysis between price, size, and age.
- Location-specific price insights.
- Segmentation of properties into value vs premium markets.

---

## Dataset
- **Type**: Synthetic housing dataset generated for demonstration purposes.
- **Size**: 1000 records.
- **Features**:
  - `square_feet` – Property size in square feet.
  - `bedrooms` – Number of bedrooms (1–6).
  - `age_years` – Age of the property (1–50 years).
  - `price` – Property price ($200k–$800k), adjusted for location.
  - `location` – Categorical: Downtown, Suburbs, Rural.

---

## Dependencies
- Python 3.8+
- [Marimo](https://github.com/marimo-framework/marimo)
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/housing-analysis-notebook.git
cd housing-analysis-notebook

# Install dependencies
pip install marimo pandas numpy matplotlib seaborn scikit-learn
