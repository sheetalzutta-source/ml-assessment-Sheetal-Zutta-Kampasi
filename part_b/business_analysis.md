
# Part B: Business Case Analysis

## B1. Problem Formulation

### B1(a) — Machine Learning Problem Formulation

This is a **multi-class classification** problem. The target variable is 
`promotion_type` — the specific promotion to deploy for each store each month 
(Flat Discount, BOGO, Free Gift with Purchase, Category-Specific Offer, or 
Loyalty Points Bonus). The candidate input features are store size, location 
type, monthly footfall, local competition density, and customer demographics. 
The goal is to predict which promotion maximises items sold.

### B1(b) — Why Items Sold is a Better Target Variable

Items sold (sales volume) is a more reliable target variable than total sales 
revenue because revenue is affected by price variations across promotions, 
making it inconsistent. Items sold directly reflects customer response to a 
promotion regardless of price. This illustrates the broader principle that the 
target variable should directly measure the business outcome being optimised, 
not a proxy that introduces confounding factors.

### B1(c) — Alternative Modelling Strategy

Instead of one global model across all 50 stores, a better strategy would be 
to build **store-level or cluster-level models** — grouping stores by location 
type (urban, semi-urban, rural) and training separate models for each group. 
This accounts for the fact that stores in different locations respond very 
differently to the same promotion, reducing noise and improving prediction 
accuracy for each store type.

## B2. Data and EDA Strategy

### B2(a) — Joining Tables and Dataset Grain

The four tables can be joined as follows:
- Join **transactions** with **store attributes** on `store_id`
- Join with **promotion details** on `promotion_type`
- Join with **calendar** on `transaction_date` to get `is_weekend` and `is_festival`

The grain of the final modelling dataset should be **one row per store per month**, 
with aggregated metrics such as total items sold, most common promotion used, 
and average competition density for that period.

### B2(b) — EDA Analyses

Four key analyses to perform before modelling:

1. **Target distribution** — Plot the distribution of items sold across stores 
to check for skewness or outliers that may affect model performance.

2. **Promotion vs items sold** — Box plots of items sold for each promotion type 
to understand which promotions generally perform better.

3. **Correlation heatmap** — Check correlations between numerical features 
(store size, competition density, footfall) and items sold to identify 
the most predictive features.

4. **Location type analysis** — Compare average items sold across urban, 
semi-urban, and rural stores to understand how location affects promotion 
effectiveness.

### B2(c) — Class Imbalance

If 80% of transactions have no promotion, the model may become biased towards 
predicting "no promotion" as it is the most frequent class. To address this:
- Use **oversampling** (e.g. SMOTE) on minority promotion classes
- Use **class weights** in the model to penalise misclassification of minority classes
- Evaluate using **F1-score** rather than accuracy, as accuracy is misleading 
with imbalanced data

## B3. Model Evaluation and Deployment

### B3(a) — Train-Test Split and Evaluation Metrics

A random split is inappropriate here because the data is time-ordered — 
randomly splitting would allow the model to train on future data and test 
on past data, causing data leakage and overly optimistic results.

Instead, use a **temporal split** — train on the first 2 years of data 
and test on the most recent year. This simulates real-world deployment 
where the model predicts future months based on past patterns.

Evaluation metrics:
- **Accuracy** — overall percentage of correct promotion recommendations
- **F1-score (macro)** — balances precision and recall across all 5 promotion 
types, important given potential class imbalance
- **Confusion matrix** — identifies which promotions are most commonly confused 
with each other, helping the marketing team understand model errors

### B3(b) — Feature Importance and Different Recommendations

To investigate why the model recommends Loyalty Points Bonus for Store 12 
in December but Flat Discount in March:
- Extract **feature importances** from the model to identify which features 
drive the recommendation
- Compare the feature values for Store 12 in December vs March — for example, 
December may have higher footfall and festival flags, making loyalty points 
more effective, while March may have higher competition density favouring 
flat discounts
- Communicate this to the marketing team by showing a **feature contribution 
chart** (e.g. SHAP values) for each month, explaining in plain language 
which store conditions led to each recommendation

### B3(c) — End-to-End Deployment Process

**Saving the model:**
Save the trained pipeline using `joblib.dump(model, 'promotion_model.pkl')` 
so it can be reloaded without retraining.

**Feeding new data:**
At the start of each month, prepare the new monthly data for all 50 stores 
in the same format as training data — applying the same preprocessing pipeline 
— and run `model.predict()` to generate recommendations.

**Monitoring for degradation:**
- Track **prediction confidence** over time — a drop may indicate the model 
is uncertain about new patterns
- Compare **actual vs predicted** promotion performance each month and 
calculate rolling RMSE or F1-score
- Set a threshold — if performance drops below a defined level for 
2-3 consecutive months, trigger retraining with fresh data
- Monitor for **data drift** — check if the distribution of input features 
(e.g. competition density, footfall) has shifted significantly from 
the training data

