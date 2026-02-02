# Dynamic Pricing Strategy for a Ride-Sharing Service

This project implements a dynamic pricing strategy for a ride-sharing service, similar to Uber or Lyft. The goal is to adjust ride prices based on real-time supply and demand, along with other factors, to maximize revenue and profitability.

## How to Use

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Jupyter Notebook:**
    Open and run the `Dynamic_pricing.ipynb` notebook in a Jupyter environment.

## Methodology

### 1. Data Preprocessing

A data preprocessing pipeline is implemented in the `data_preprocessing_pipeline` function. The key steps are:

-   **Handling Missing Values:**
    -   Numeric features: Missing values are filled with the mean of the respective column.
    -   Categorical features: Missing values are filled with the mode of the respective column.
-   **Outlier Detection and Handling:**
    -   Outliers in numeric features are detected using the Interquartile Range (IQR) method.
    -   Values outside the 1.5 * IQR range are replaced with the mean of the feature.
-   **Feature Engineering:**
    -   The `Vehicle_Type` categorical feature is converted into a numerical feature:
        -   `Premium`: 1
        -   `Economy`: 0

### 2. Machine Learning Model

A **Random Forest Regressor** (`sklearn.ensemble.RandomForestRegressor`) is used to predict the `adjusted_ride_cost`.

The model is trained on the following features:
-   `Number_of_Riders`
-   `Number_of_Drivers`
-   `Vehicle_Type`
-   `Expected_Ride_Duration`

### 3. Prediction

The notebook provides a `predict_price` function to predict the ride cost based on user inputs.

**To make a prediction:**

1.  Modify the following user input values in the last cell of the notebook:
    ```python
    user_number_of_riders = 50
    user_number_of_drivers = 25
    user_vehicle_type = "Economy"
    Expected_Ride_Duration = 30
    ```
2.  Run the cell to get the predicted price.

## Libraries Used

-   pandas
-   plotly
-   numpy
-   scikit-learn
