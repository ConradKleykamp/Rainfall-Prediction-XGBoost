# Rainfall-Prediction-XGBoost
Predicting whether or not it will rain with a XGBoost model

![image](https://github.com/user-attachments/assets/c1813981-1443-46d0-b3be-dad0fab2f404)

---

### Objective
This project has been completed as part of Kaggle's Playground Series competitions (Season 5, Episode 3). The overarching goal of this project is to develop a model to predict whether or not there will be rainfall for each day of the year.

In this project, I will first briefly explore the datasets by viewing basic summary statistics and visualizing the distributions of the target variable and the numerical features. I will also explore potential correlation/interaction amongst features via the use of heatmaps, pair plots, and violin plots.

Following the brief EDA, I will employ adversarial validation to determine whether or not the original data (discussed in the 'About the Data' section) follows the same distribution as Kaggle's synthetic data. This step will likely be a key factor in the final model's performance, as incorporating data from a different distribution can sometimes negatively impact a model's performance on the withheld testing data.

In order to prepare the data for modeling, I will impute missing values with column means, remove outliers using the Inter Quartile method, and scale the data using a MinMax scaler.

The final model will be an XGBoost model. XGBoost (XGB) is an optimized distributed gradient boosting algorithm that leverages parallel tree boosting to solve a majority of data science tasks. In the case of this project, the overall task is binary classification.

In order to identify the best model, I will use optuna, which is a hyperparameter optimization framework. Optuna will create and run through 100 trials to determine the ideal hyperparameters for the final LGBM model.

The final model of this project will be evaluated using the area under the curve (AUC) of the receiver operating characteristic curve (ROC curve). The ROC curve is esentially the plot of the true positive rate versus the false positive rate. In the case of this model and competition, a 'perfect' model would yield an AUC of 1.0, while a terrible model would likely result in an AUC of 0.5 (basically random guess). However, obtaining an AUC of 1 is typically impossible, so the objective will be to get the AUC as close to 1.0 as possible.

---

### Methods
Libraries Used
- pandas
- numpy
- matplotlib
- seaborn
- catboost
- sklearn
- xgboost
- optuna
- lightgbm

Adversarial Validation
- Determining whether or not the synthetic training data and original data are from the same distribution
- Combining both datasets, labeling all train data as 0, all original data as 1
- Leveraging a simple Catboost classifier to distinguish between the two datasets
- Metric: AUC-ROC, if close to 0.5, the datasets likely come from the same distribution

Exploratory Data Analysis
- Creating a function to show summary statistics of training and testing datasets
- Visualizing the distributions of target variable with countplot and donutplot
- Visualizing the distributions of numerical variables with histogram and KDE plots
- Visualizing potential correlations amongst numerical variables with correlation heatmap
- Visualizing interactions between target variable and several features with violin plots

Data Preprocessing
- Combining original dataset with Kaggle-provided data
- Filling in missing values with mean values of column
- Removing outliers based on IQR
- Feature engineering
- Feature scaling with MinMaxScaler

Model Building/Tuning
- Predetermined parameters:
    - Objective: binary:logistic
    - Device: cpu
    - Metic: auc
    - Random_state: 42
- Parameters to be determined:
    - N_estimators
    - Learning_rate
    - Colsample_bytree
    - Max_depth
    - Subsample
    - Min_child_weight
    - Reg_alpha
    - Reg_lambda
- 100 Optuna trials

Final Model
- Parameters:
    - N_estimators: 727
    - Learning_rate: 0.02065809145104903
    - Colsample_bytree: 0.6844525799009806
    - Max_depth: 1
    - Subsample: 0.519006802168094
    - Min_child_weight: 5
    - Reg_alpha: 0.00015480424761672249
    - Reg_lambda: 0.2256543295563229

---

### General Results
![image](https://github.com/user-attachments/assets/69b39af2-e76f-434a-9245-6305185b9e8b)
![image](https://github.com/user-attachments/assets/7e9db045-0ed3-4625-855e-b24a0eb403e1)


