# [Machine Learning] Churn Prediction & Segmentation

# I. Introduction
This project uses two machine learning approaches for different purposes: Supervised learning to predict which customers are likely to churn, and Unsupervised learning to segment customers who have churned.

# II. Dataset
Labeled data of an eCommerce store, each row represents a single customer and their information:
- CustomerID: Unique customer ID
- Churn: Churn flag (1:Churned; 0:No churn)
- Tenure: Tenure of customer in organization
- PreferredLoginDevice: Preferred login device of customer
- CityTier: Region code
- WarehouseToHome: Distance in between warehouse to home of customer
- PreferPayment: The payment method that user usually use
- Gender: Gender of customer
- HourSpendOnApp: Number of hours spend on mobile application or website
- NumberOfDeviceRegistered: Total number of devices is registered on particular customer
- PreferedOrderCat: Preferred order category of customer in last month
- SatisfactionScore: Satisfactory score of customer on service
- MaritalStatus: Marital status of customer
- NumberOfAddress: Total number of added added on particular customer
- Complain: Any complaint has been raised in last month (1:Yes; 0:No)
- OrderAmountHikeFromlastYear: Order amount hike from last year
- CouponUsed: Number of coupons customer has used
- OrderCount: Number of orders
- DaySinceLastOrder: Number of days since last order
- CashbackAmount: Total cashback amount

# III. Exploring the dataset
- [See full notebook](https://github.com/DTN-492/ML_Churn_Predict_and_Segmentation/blob/main/notebooks/00_data_process.ipynb)
- First look:
<img width="661" height="507" alt="image" src="https://github.com/user-attachments/assets/b1c9f8a0-54e0-4288-87c6-cea462784458" />

- Numeric columns summary:
<img width="667" height="672" alt="image" src="https://github.com/user-attachments/assets/71efdd16-fb0c-4d31-b3b3-0c384bfad93b" />

- Non-numeric columns summary:
<img width="755" height="312" alt="image" src="https://github.com/user-attachments/assets/d7a09ced-6a3f-434c-a4d1-b839ed8aa45f" />

- NA values map:
<img width="1292" height="592" alt="image" src="https://github.com/user-attachments/assets/71aebe7a-8fd8-4313-aef1-1d0b2ab1d64c" />

- With numeric columns, I replaced missing values with the column median
<img width="690" height="510" alt="image" src="https://github.com/user-attachments/assets/16bd5252-b786-49e8-bc9d-e4468ab7ecc7" />

- With non-numeric columns, I replace values with similar meaning into one single value only (ex: "CC" and "Credit Card" -> "CC")
<img width="587" height="317" alt="image" src="https://github.com/user-attachments/assets/4ba21f16-c32d-432b-bda5-9c168015946e" />

# IV. Visualization
- [See full notebook](https://github.com/DTN-492/ML_Churn_Predict_and_Segmentation/blob/main/notebooks/01_eda.ipynb)
- First, I used the one-hot encoding method to transform non-numeric features:
```python
df = pd.get_dummies(df, drop_first = True)
```
- In order to determine which features are important, I fit 70% of the dataset with the Random Forest model and extract features with importance score above 0.05
  + Model performance on test set:
<img width="500" height="186" alt="image" src="https://github.com/user-attachments/assets/574e8eac-2dcc-4b3f-bef0-5dccc5cada99" />

  + Importance score:
<img width="1012" height="532" alt="image" src="https://github.com/user-attachments/assets/0905436f-1b1e-41fe-af2f-5f86eaa23fcf" />

- Compare distribution within 2 groups: Churn and No churn with important features above
<img width="1511" height="811" alt="image" src="https://github.com/user-attachments/assets/5f2e377b-ecc5-4f30-8dad-b7fabfa26a94" />
<img width="567" height="433" alt="image" src="https://github.com/user-attachments/assets/85f7ec34-7abb-4bad-8a63-8f28599d3cd8" />

- Churn rate of different groups:
<img width="1610" height="1011" alt="image" src="https://github.com/user-attachments/assets/9d524101-c6f2-4faf-830f-1dfe965ce6f8" />

- ***Key takaways:***
  - Insights:
    + New users tend to churn more
    + Users with lower Cashback Amount are more likely to churn more
    + Customers with high number of addresses & long time no order have higher chance of churn
  - Suggestions:
    + Focus on new customers and customers with last order > 1 month, launch cashback promotions for them to acquire/retain
    + Aware of customers with high number of addresses, limit the number of addresses can be added or implement verify identity in order to reduce the risk of fraud

# V. Building prediction model
- [See full notebook](https://github.com/DTN-492/ML_Churn_Predict_and_Segmentation/blob/main/notebooks/02_fit_model.ipynb)
- In this section, I split the dataset into 3 parts: train/valid/test with 70/10/20 ratio
- The dataset is quite imbalanced, so I adjust the class weight hyperparameters for better performance and avoid bias
<img width="305" height="95" alt="image" src="https://github.com/user-attachments/assets/0a198249-832f-46b2-8fbb-a9c8c7fb7cac" />

```python
churn_wt = y_train.count()/y_train.sum() - 1 # y_train contains Target feature (Churn) only
all_mod = {"Random Forest"  : RandomForestClassifier(random_state = 10, class_weight = {0:1, 1:churn_wt}),
           "Hist Gradient Boost" : HistGradientBoostingClassifier(random_state = 10, class_weight = {0:1, 1:churn_wt}),
           "XGBoost" : XGBClassifier(random_state = 10, scale_pos_weight = churn_wt)
            }
```
- After fitting 3 models on the training set, here are performance metrics on the valid set:
<img width="835" height="357" alt="image" src="https://github.com/user-attachments/assets/2e96cc05-49fe-40d1-b3a4-8be010a6a46b" />

- The XGBoost model performs extremely well, so I tune this model using the Optuna library and here are the result on the test set after tuning:
<img width="507" height="180" alt="image" src="https://github.com/user-attachments/assets/9e587134-c6cb-4661-a53c-6b2aa52d8b86" />

- ROC curves:
<img width="459" height="469" alt="image" src="https://github.com/user-attachments/assets/cd08861e-b5b1-4cc5-a882-745378cda9c8" />

- SHAP values:
<img width="864" height="498" alt="image" src="https://github.com/user-attachments/assets/0d79f546-4871-423b-89ba-c9c6232eab2e" />

- ***Key takaways:***
  + As usual, XGBoost work extremely well on tabular dataset
  + Base on SHAP values, we can easily explain machine learning predictions by measuring how much each input feature pushes the model's output up or down

# VI. Segmentation
- [See full notebook](https://github.com/DTN-492/ML_Churn_Predict_and_Segmentation/blob/main/notebooks/03_clustering.ipynb)
- In this section, I use K-means model to separate Churned customer
- To reduce the dimension, I use PCA with 4 components (explain ~ 82% of variance)
<img width="556" height="413" alt="image" src="https://github.com/user-attachments/assets/ded4b5ed-2ddb-4ddd-ae64-9805f595814f" />

- Since K-means model sensitive to outliers, scaling is an important preprocessing step, and here is the WCSS chart after fitting the K-means model on the scaled PCA dataset:
<img width="851" height="469" alt="image" src="https://github.com/user-attachments/assets/94962db7-5ecf-4105-aa65-b54b851f6cda" />

-> We choose k = 2
