# Gold Price Prediction using Machine Learning 📈

## 🎯 Project Overview
This project focuses on **predicting gold prices** using various financial market indicators through supervised machine learning algorithms. Gold price prediction is crucial for investors, financial analysts, and economists to make informed decisions in volatile markets. The project implements multiple regression models to forecast gold prices based on correlated financial instruments.

## 📊 Objectives
* **Analyze** the relationship between gold prices and other financial indicators
* **Preprocess** financial time series data and handle datetime features
* **Build and Train** multiple machine learning models for price prediction
* **Compare** model performance using R² score and Mean Squared Error
* **Optimize** hyperparameters using GridSearchCV for best performance
* **Visualize** correlations and model predictions

## 🛠️ Tech Stack
* **Programming Language:** Python
* **Libraries:**
    * `pandas`, `numpy` (Data Manipulation)
    * `matplotlib`, `seaborn` (Data Visualization)
    * `scikit-learn` (Machine Learning: Preprocessing, Model Training, Evaluation)
    * `xgboost` (Gradient Boosting Algorithm)
    * `warnings` (Warning Management)

## 📁 Dataset
* **File:** `gold_price_data.csv` (included in this repository)
* **Description:** The dataset contains daily financial market data with 2,290 entries, including:
    * `Date`: Timestamp (converted to datetime format)
    * `SPX`: S&P 500 Index value
    * `GLD`: Gold Price (target variable)
    * `USO`: United States Oil Fund price
    * `SLV`: Silver Price (dropped due to high correlation)
    * `EUR/USD`: Euro to US Dollar exchange rate
* **Goal:** Predict gold prices based on correlated financial indicators (supervised learning)

## 🔬 Methodology & Steps
1.  **Data Loading & Initial Exploration:**
    * Loaded the `gold_price_data.csv` dataset using pandas with date parsing.
    * Inspected data structure using `.info()` and checked for missing values.
    * Verified no null values in the dataset using `.isna().sum()`.
2.  **Data Preprocessing & Feature Engineering:**
    * **Correlation Analysis:** Generated correlation matrix heatmap to identify feature relationships.
    * **Feature Selection:** Dropped `SLV` column due to high correlation (0.866) with target variable `GLD`.
    * **Time Series Handling:** Set `Date` column as index for time-based analysis.
3.  **Exploratory Data Analysis (EDA):**
    * Visualized `EUR/USD` price trends over time using line plots.
    * Analyzed correlation patterns between different financial indicators.
    * Identified strong positive correlation between `GLD` and `SPX` (0.040).
4.  **Data Preparation for Modeling:**
    * **Feature-Target Separation:** Split data into features (`SPX`, `USO`, `EUR/USD`) and target (`GLD`).
    * **Data Scaling:** Applied `StandardScaler` to normalize feature values.
    * **Train-Test Split:** Divided data into training and testing sets.
5.  **Model Building & Training:**
    * **Lasso Regression with Polynomial Features:** Pipeline with polynomial feature transformation and L1 regularization.
    * **Random Forest Regressor:** Ensemble method with multiple decision trees.
    * **XGBoost Regressor:** Gradient boosting algorithm known for high performance.
    * **Hyperparameter Tuning:** Used `GridSearchCV` for optimal parameter selection.
6.  **Model Evaluation & Comparison:**
    * **Evaluation Metrics:** R² Score and Mean Squared Error (MSE).
    * **Performance Comparison:** Compared all three models on test data.
    * **Best Model Selection:** Identified the most accurate prediction model.

## 📈 Results
* The correlation analysis revealed key relationships between gold prices and financial indicators.
* Multiple regression models were successfully implemented and evaluated.
* Model performance was quantitatively compared using standard regression metrics.
* The project provides a robust framework for financial price prediction.

## 🚀 How to Run
1.  **Clone this repository:**
    ```bash
    git clone https://github.com/hvlr2111/Gold_Price_Prediction.git
    cd gold-price-prediction
    ```
2.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook Gold_Price_Prediction.ipynb
    ```
4.  **Execute the notebook cells sequentially:**
    * Ensure `gold_price_data.csv` is in the same directory.
    * Follow the data preprocessing steps.
    * Run model training and evaluation cells.
    * Analyze the results and visualizations.

## 📝 Key Features
* **Comprehensive Data Analysis:** Correlation heatmaps and time series visualization
* **Multiple Algorithm Implementation:** Three different regression approaches
* **Hyperparameter Optimization:** Automated tuning for best performance
* **Model Comparison:** Quantitative evaluation of prediction accuracy
* **Financial Market Insights:** Understanding gold price drivers

## 🔮 Future Work
* **Time Series Analysis:** Implement ARIMA, LSTM, or other time series models.
* **Feature Engineering:** Create additional technical indicators and lag features.
* **Real-time Prediction:** Develop API for live gold price forecasting.
* **Portfolio Integration:** Extend to portfolio optimization and risk management.
* **Additional Features:** Incorporate macroeconomic indicators and news sentiment.

## 👤 Author
* H.V.L.Ranasinghe
* LinkedIn: `https://www.linkedin.com/in/lakshika-ranasinghe-1404ab34a/`
