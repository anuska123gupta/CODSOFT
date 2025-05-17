\'\'\'python
# CODSOFT
Repository for CODSOFT Data Science Internship tasks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer

# --- 1. Load the Dataset ---
DATASET_PATH = 'cleaned_movie_data.csv'  # **REPLACE WITH YOUR ACTUAL PATH**

try:
    df = pd.read_csv(DATASET_PATH)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print(f"Error: Dataset not found at {DATASET_PATH}. Please check the file path.")
    exit()

# --- 2. Exploratory Data Analysis (EDA) ---
print("\n--- Exploratory Data Analysis ---")
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset info:")
df.info()
print("\nSummary statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

#  CHECK THE COLUMN NAME AND REPLACE IF NECESSARY
rating_column_name = 'Rating'  #  <--- CHANGE THIS IF YOUR RATING COLUMN IS NAMED DIFFERENTLY

# Visualize the target variable (rating) distribution
plt.figure(figsize=(8, 6))
sns.histplot(df[rating_column_name], bins=30, kde=True)  # Use the correct column name
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

# --- 3. Feature Engineering ---
print("\n--- Feature Engineering ---")

# **Identify your target variable and features based on your dataset**
TARGET_COLUMN = rating_column_name  #  Use the corrected name
FEATURE_COLUMNS = [col for col in df.columns if col != TARGET_COLUMN]

# **Identify categorical and numerical features accurately**
CATEGORICAL_FEATURES = [col for col in FEATURE_COLUMNS if df[col].dtype == 'object']
NUMERICAL_FEATURES = [col for col in FEATURE_COLUMNS if df[col].dtype != 'object']

print("\nCategorical features:", CATEGORICAL_FEATURES)
print("Numerical features:", NUMERICAL_FEATURES)

# --- 4. Data Preprocessing ---
print("\n--- Data Preprocessing ---")

# Create transformers for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Handle missing numerical values
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing categorical values
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, NUMERICAL_FEATURES),
        ('cat', categorical_transformer, CATEGORICAL_FEATURES)
    ],
    remainder='passthrough'  # Drop other columns if any
)

# --- 5. Model Selection and Training ---
print("\n--- Model Selection and Training ---")

# Split data into training and testing sets
X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]  # Use the corrected name
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define different models to try
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    results[name] = {'MSE': mse, 'RMSE': rmse, 'R-squared': r2, 'MAE': mae}

# Print the initial results
print("\nInitial Model Evaluation:")
for name, metrics in results.items():
    print(f"\n{name}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

# --- 6. Hyperparameter Tuning (Optional but Recommended) ---
print("\n--- Hyperparameter Tuning (Optional) ---")

# Example of hyperparameter tuning for Random Forest
param_grid_rf = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [None, 5, 10],
    'regressor__min_samples_split': [2, 5, 10]
}

grid_search_rf = GridSearchCV(Pipeline(steps=[('preprocessor', preprocessor),
                                             ('regressor', RandomForestRegressor(random_state=42))]),
                             param_grid_rf,
                             cv=3,  # 3-fold cross-validation
                             scoring='neg_mean_squared_error',
                             n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

print("\nBest parameters for Random Forest:", grid_search_rf.best_params_)
best_rf_model = grid_search_rf.best_estimator_
y_pred_best_rf = best_rf_model.predict(X_test)
mse_best_rf = mean_squared_error(y_test, y_pred_best_rf)
rmse_best_rf = np.sqrt(mse_best_rf)
r2_best_rf = r2_score(y_test, y_pred_best_rf)
mae_best_rf = mean_absolute_error(y_test, y_pred_best_rf)

print("\nBest Random Forest Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse_best_rf:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_best_rf:.4f}")
print(f"R-squared (R²): {r2_best_rf:.4f}")
print(f"Mean Absolute Error (MAE): {mae_best_rf:.4f}")

# --- 7. Model Evaluation and Interpretation ---
print("\n--- Model Evaluation and Interpretation ---")

# You can further analyze the performance of the best model:
# - Residual plots to check for patterns in errors
# - Feature importance (for tree-based models like Random Forest and Gradient Boosting)

# Example of feature importance for Random Forest
if 'Random Forest' in models:
    try:
        feature_importances = best_rf_model.named_steps['regressor'].feature_importances_
        feature_names = preprocessor.transformers_[0][2] + \
            list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(CATEGORICAL_FEATURES))

        sorted_indices = np.argsort(feature_importances)[::-1]

        plt.figure(figsize=(12, 6))
        plt.title("Feature Importances (Random Forest)")
        plt.bar(range(X_train.shape[1]), feature_importances[sorted_indices], align="center")
        plt.xticks(range(X_train.shape[1]), np.array(feature_names)[sorted_indices], rotation=90)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Could not plot feature importances: {e}")

# --- 8. Saving the Model (Optional) ---
# import joblib
# best_model = best_rf_model # Or your chosen best model
# joblib.dump(best_model, 'movie_rating_prediction_model.pkl')
# print("\nBest model saved as movie_rating_prediction_model.pkl")
\'\'\'
