# Predicting Supply Chain Disruption in Life Science Manufacturing Industry

This project develops a **decision support system (DSS)** to predict and explain **supply chain disruptions** (delivery delays) in the **life science manufacturing industry**. Using historical pharmaceutical shipment data and a machine learning model, it estimates the risk of delay for new shipments and exposes insights via a **Streamlit dashboard**.

The work was created as a **DSS Final Project**.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Data & Problem Description](#data--problem-description)  
- [Methodology](#methodology)  
- [Repository Structure](#repository-structure)  
- [Modeling Approach](#modeling-approach)  
- [Streamlit Dashboard](#streamlit-dashboard)  
- [How to Run the Project](#how-to-run-the-project)  
- [Results & Insights](#results--insights)  
- [Future Work](#future-work)  
- [License](#license)

---

## Project Overview

Supply chain disruptions in the **life science sector** can have serious implications for both **public health** and **business performance**. Delayed shipments of pharmaceutical products can lead to stockouts, regulatory issues, and patient impact.

This project:

- Uses **real-world pharmaceutical shipment data**  
- Analyzes **drivers of delivery delays**  
- Trains a **classification model** to predict delay risk  
- Provides a **user-friendly Streamlit app** for supply chain planners to:
  - Explore historical patterns
  - Estimate delay risk for future/planned shipments
  - View explanations in intuitive risk categories (Low / Medium / High)

---

## Data & Problem Description

The dataset contains **10,000+ shipment records** from various pharmaceutical companies delivering products to multiple countries.

Typical columns (may vary depending on raw file):

- Shipment & order information:  
  - `po_date`, `scheduled_delivery_date`, `actual_delivery_date`  
  - `shipment_mode` (e.g., air, sea, road)  
  - `country` / `destination_country`  
  - `vendor`, `carrier`, `incoterm`  
- Product & cost information:  
  - `product_group`, `material`  
  - `quantity`, `weight`, `volume`, `unit_price`, `total_cost`  
- Target variable:  
  - **Delayed vs On-time** shipment (e.g., `delayed` = 1 if delivered after scheduled date, else 0)

**Objective:**  
Given shipment details **known at planning time**, predict whether a shipment is **likely to be delayed** and quantify its **risk level**.

---

## Methodology

The project follows a standard **data science lifecycle**:

1. **Data Cleaning**
   - Parse date columns into proper datetime format  
   - Convert numeric-like object columns (quantities, costs) to numeric types  
   - Handle missing values and invalid records  
   - Filter out extreme outliers and records missing critical dates

2. **Feature Engineering**
   - Delivery delay:
     - `delay_days = actual_delivery_date - scheduled_delivery_date`
   - Lead times:
     - `po_to_schedule = scheduled_delivery_date - po_date`  
     - `po_to_delivery = actual_delivery_date - po_date`
   - Categorical encodings:
     - One-hot or label encoding for `country`, `vendor`, `shipment_mode`, `product_group`, etc.
   - Cleanup / scaling of numeric features (if needed)

3. **Exploratory Data Analysis (EDA)**
   - Delay rates by:
     - Country / region  
     - Shipment mode  
     - Vendor / carrier  
     - Product group
   - Distribution of delay days  
   - Identification of **“risk hotspots”**

4. **Machine Learning Modeling**
   - Train a **Random Forest classifier** to predict “Delayed vs On-time”  
   - Use engineered lead-time and shipment features as inputs  
   - Output: probability of delay for each shipment

5. **Decision Support Interface (Streamlit)**
   - An interactive dashboard to:
     - Visualize historical delay patterns  
     - Estimate risk for user-specified planned shipments  
     - Map model probabilities to **Low / Medium / High** risk levels

---

## Repository Structure

> Adjust filenames if they differ in your actual `my final project` folder.
```text
SC_LS_disruption_DSS/
├─ my final project/
│  ├─ data/
│  │  ├─ raw/                # Original dataset(s)
│  │  └─ processed/          # Cleaned / feature-engineered data
│  ├─ notebooks/
│  │  ├─ 01_eda.ipynb        # Exploratory data analysis
│  │  ├─ 02_feature_eng.ipynb# Feature engineering experiments
│  │  └─ 03_modeling.ipynb   # Model training & evaluation
│  ├─ src/
│  │  ├─ data_preparation.py # Data cleaning & feature engineering functions
│  │  ├─ train_model.py      # Script to train & persist the Random Forest model
│  │  └─ utils.py            # Helper utilities (metrics, plotting, etc.)
│  ├─ app/
│  │  └─ streamlit_app.py    # Main Streamlit dashboard
│  ├─ models/
│  │  └─ random_forest.pkl   # Saved trained model (and encoders/scalers)
│  ├─ requirements.txt       # Python dependencies
│  └─ README.md              # (Optional) Module-level readme
└─ README.md                 # Root project readme (this file)
