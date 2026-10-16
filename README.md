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

- Key takaways:
  - Insights:
    + New users tend to churn more
    + Users with lower Cashback Amount are more likely to churn more
    + Customers with high number of addresses & long time no order have higher chance of churn
  - Suggestions:
    + Focus on new customers and customers with last order > 1 month, launch cashback promotions for them to acquire/retain
    + Aware of customers with high number of addresses, limit the number of addresses can be added or implement verify identity in order to reduce the risk of fraud

