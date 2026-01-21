Predicting Supply Chain Disruption in Life Science Manufacturing Industry
Abstract
This project aims to develop a decision support system that predicts supply chain disruptions in the life science manufacturing industry. By leveraging historical shipment data, the system identifies factors contributing to delivery delays and provides actionable insights to supply chain planners. A machine learning model is trained to classify shipments as likely delayed or on-time, and a user-friendly Streamlit dashboard is developed to visualize key patterns and estimate risk for future shipments.
Introduction
Supply chain disruptions in the life science sector can have significant impacts on public health and business operations. This project uses real-world pharmaceutical shipment data to analyze delay patterns, build predictive models, and create a dashboard that supports proactive decision-making.
Data Description
The dataset used contains over 10,000 shipment records from various pharmaceutical companies supplying products to different countries. Key columns include shipment dates, country, vendor, shipment mode, product group, quantities, costs, and more. The target variable is whether a shipment was delayed (delivered after the scheduled date).
Methodology
The project follows these steps:
1. Data Cleaning: Handle missing values, convert dates, and clean numeric columns.
2. Feature Engineering: Derive features like delivery delay days, lead times, and encode categorical variables.
3. Exploratory Data Analysis (EDA): Visualize delay rates by country, shipment mode, vendor, and product group.
4. Machine Learning Modeling: Train a Random Forest classifier to predict shipment delays.
5. Streamlit Dashboard: Develop an interactive dashboard for EDA and shipment risk estimation.
Data Cleaning
Data cleaning involved parsing date columns, converting object columns with numeric data, and handling missing or invalid entries. Shipments with missing critical dates or extreme outliers were excluded from modeling.
Feature Engineering
Features engineered include:
- Delivery delay in days (Delivered - Scheduled)
- Lead time features (PO-to-Schedule, PO-to-Delivery)
- Encoded categorical variables (country, mode, vendor, etc.)
- Cleaned numeric fields (quantities, costs, weights)
Exploratory Data Analysis (EDA)
EDA revealed significant variation in delay rates by country, shipment mode, and vendor. Visualization of delay distributions and summary tables highlighted risk hotspots in the supply chain.
Modeling
A Random Forest classifier was trained to predict if a shipment would be delayed. Model features included engineered lead times, shipment characteristics, and encoded categorical variables. The model output is a probability of delay, which is mapped to Low/Medium/High risk levels for end users.
Dashboard
A Streamlit dashboard was built to make insights accessible and actionable. Key functionalities include:
- Visualization of delay rates and risk hotspots
- Interactive shipment risk estimation based on user input
- Explanations for risk predictions based on historical patterns
The dashboard is designed for business users and does not expose technical model metrics.
Results and Insights
The system provides clear visibility into which countries, shipment modes, and vendors have higher risk of delay. Users can proactively assess planned shipments and receive understandable risk levels (Low/Medium/High) with explanations. This supports better planning and mitigation of supply chain disruptions in the life science sector.
Conclusion
This project demonstrates the value of combining data-driven analysis, machine learning, and user-centric dashboards for supply chain risk management in the life science manufacturing industry. The approach can be extended to other sectors and incorporate more real-time data and advanced models for even greater impact.
