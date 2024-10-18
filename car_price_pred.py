import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from matplotlib.ticker import FuncFormatter
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('cardekho.csv')

print("Few Rows of Data -- ")
print(df.head())
print("\nInfo of Data -- ")
print(df.info())

# Show the shape of the dataset
print("\nShape of the Data -- ")
print(df.shape)

print("\nDescription of Dataset")
print(round(df.describe(),2).T)


# finding missing values - 
print("\nMissing values in Data -- ")
print(df.isnull().sum())

# Drop rows with missing values
df_cleaned = df.dropna()

# Display the number of rows before and after dropping missing values
print(f'\nOriginal number of rows: {df.shape[0]}')
print(f'Number of rows after dropping missing values: {df_cleaned.shape[0]}')

# Display the cleaned dataset
print(df_cleaned.head())
# Check for any remaining missing values
print(df_cleaned.isnull().sum())

# Check for duplicate rows
duplicate_rows = df.duplicated()

# Display the number of duplicate rows
num_duplicates = duplicate_rows.sum()
print(f'\nNumber of duplicate rows: {num_duplicates}')

 #display the duplicate rows
if num_duplicates > 0:
    print("\nDuplicate Rows -- ")
    print(df[duplicate_rows])

# Drop duplicate rows
df_cleaned = df.drop_duplicates()

# Display the number of rows before and after dropping duplicates
print(f'\nOriginal number of rows: {df.shape[0]}')
print(f'Number of rows after dropping duplicates: {df_cleaned.shape[0]}')

# Check for any remaining duplicate rows
remaining_duplicates = df_cleaned.duplicated().sum()
print(f'Number of remaining duplicate rows: {remaining_duplicates}')


# handling data types
# Remove 'bhp' and spaces from 'max_power'
df_cleaned['max_power'] = df_cleaned['max_power'].replace({'bhp': '', ' ': ''}, regex=True)

# Convert 'max_power' to numeric, coercing errors to NaN
df_cleaned['max_power'] = pd.to_numeric(df_cleaned['max_power'], errors='coerce')

# Handle NaN values in 'seats' column before conversion
df_cleaned['seats'].fillna(df_cleaned['seats'].mean(), inplace=True)  # You can choose to fill with mode or drop NaN

# Convert 'seats' to integer
df_cleaned['seats'] = df_cleaned['seats'].astype(int)

# Convert categorical columns to category dtype
df_cleaned['fuel'] = df_cleaned['fuel'].astype('category')
df_cleaned['seller_type'] = df_cleaned['seller_type'].astype('category')
df_cleaned['transmission'] = df_cleaned['transmission'].astype('category')
df_cleaned['owner'] = df_cleaned['owner'].astype('category')

# Verify the data types after conversion
print(df_cleaned.dtypes)

# Print cleaned DataFrame to verify changes
print(df_cleaned)

# identifying and handling outliers
# Function to identify outliers using Z-score
def identify_outliers_zscore(df, column):
    z_scores = np.abs(stats.zscore(df[column]))
    return np.where(z_scores > 3)[0]  # Indices of outliers

# Capping Outliers Function
def cap_outliers(df, column):
    upper_limit = df[column].quantile(0.95)
    lower_limit = df[column].quantile(0.05)
    df[column] = np.where(df[column] > upper_limit, upper_limit, df[column])
    df[column] = np.where(df[column] < lower_limit, lower_limit, df[column])

# Columns to check for outliers
outlier_columns = ['selling_price', 'km_driven', 'max_power', 'seats']

# Convert columns to numeric and handle non-numeric values
for col in outlier_columns:
    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')  # Convert to numeric, set errors to NaN

# Optionally drop rows with NaN values in the specified columns
df_cleaned = df_cleaned.dropna(subset=outlier_columns)

# Visualize before capping outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_cleaned[outlier_columns])
plt.title('Boxplots of Numeric Columns Before Capping Outliers')
plt.show()

# Identify and cap outliers for each specified column
for col in outlier_columns:
    zscore_outliers = identify_outliers_zscore(df_cleaned, col)
    print(f'Outliers using Z-score method in {col}: {len(zscore_outliers)}')

    cap_outliers(df_cleaned, col)

# Visualize after capping outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_cleaned[outlier_columns])
plt.title('Boxplots of Numeric Columns After Capping Outliers')
plt.show()

# Final Data Overview
print("\nFinal Data Overview After Capping Outliers")
print(round(df_cleaned.describe(),2).T)

# Verify data types after all transformations
print("\nData Types After Transformations:")
print(df_cleaned.dtypes)
print(df_cleaned.shape)

# Function to format y-axis ticks in lakhs
def lakhs_formatter(x, pos):
    return f'{x/100000:.2f}L'  # Convert to lakhs

categorical_columns = ['fuel', 'seller_type', 'transmission', 'owner']

# Calculate correlation matrix only on numeric columns
correlation_matrix = df_cleaned.select_dtypes(include=[np.number]).corr()

# Print the correlation matrix
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Categorical variable analysis
for col in categorical_columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_cleaned, x=col, palette='viridis')
    plt.title(f'Count of {col}')
    plt.xticks(rotation=45)
    plt.show()

# Selling Price vs. Other Variables
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df_cleaned, x='km_driven', y='selling_price', hue='fuel', alpha=0.6)
plt.title('Selling Price vs. KM Driven')
plt.xlabel('KM Driven')
plt.ylabel('Selling Price')

# Set the y-axis formatter for selling price in lakhs
plt.gca().yaxis.set_major_formatter(FuncFormatter(lakhs_formatter))

plt.legend()
plt.show()

# Price Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned['selling_price'], bins=30, kde=True)
plt.title('Distribution of Selling Price')
plt.xlabel('Selling Price')

# Set the x-axis formatter for price distribution in lakhs
plt.gca().xaxis.set_major_formatter(FuncFormatter(lakhs_formatter))

plt.ylabel('Frequency')
plt.show()

# Boxplot of Selling Price by Fuel Type
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_cleaned, x='fuel', y='selling_price', palette='viridis')
plt.title('Selling Price Distribution by Fuel Type')
plt.ylabel('Selling Price (in Lakhs)')
plt.gca().yaxis.set_major_formatter(FuncFormatter(lakhs_formatter))
plt.xticks(rotation=45)
plt.show()

# Boxplot of Selling Price by Seller Type
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_cleaned, x='seller_type', y='selling_price', palette='viridis')
plt.title('Selling Price Distribution by Seller Type')
plt.ylabel('Selling Price (in Lakhs)')
plt.gca().yaxis.set_major_formatter(FuncFormatter(lakhs_formatter))
plt.xticks(rotation=45)
plt.show()

# Boxplot of Selling Price by Transmission
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_cleaned, x='transmission', y='selling_price', palette='viridis')
plt.title('Selling Price Distribution by Transmission')
plt.ylabel('Selling Price (in Lakhs)')
plt.gca().yaxis.set_major_formatter(FuncFormatter(lakhs_formatter))
plt.show()

# Boxplot of Selling Price by Owner
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_cleaned, x='owner', y='selling_price', palette='viridis')
plt.title('Selling Price Distribution by Owner')
plt.ylabel('Selling Price (in Lakhs)')
plt.gca().yaxis.set_major_formatter(FuncFormatter(lakhs_formatter))
plt.show()


# Pairplot to examine relationships between numeric features
numeric_columns = ['selling_price', 'km_driven', 'max_power', 'seats']
sns.pairplot(df_cleaned[numeric_columns], diag_kind='kde')
plt.suptitle('Pairplot of Numeric Features', y=1.02)
plt.show()

# Create a pivot table for the heatmap
heatmap_data = df_cleaned.groupby(['fuel', 'transmission']).size().unstack()

plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='Blues')
plt.title('Count of Cars by Fuel Type and Transmission')
plt.ylabel('Fuel Type')
plt.xlabel('Transmission')
plt.show()



# Count of cars by year of manufacture (assuming 'year' is a column)
plt.figure(figsize=(12, 6))
sns.countplot(data=df_cleaned, x='year', palette='viridis')
plt.title('Count of Cars Over Years')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned['km_driven'], bins=30, kde=True)
plt.title('Distribution of KM Driven')
plt.xlabel('KM Driven')
plt.ylabel('Frequency')
plt.show()

# Convert selling price to lakhs in the DataFrame
df_cleaned['selling_price_lakhs'] = df_cleaned['selling_price'] / 100000

# Average Selling Price by Fuel Type
plt.figure(figsize=(10, 6))
sns.barplot(data=df_cleaned, x='fuel', y='selling_price_lakhs', estimator=np.mean, palette='Set2')
plt.title('Average Selling Price by Fuel Type')
plt.xlabel('Fuel Type')
plt.ylabel('Average Selling Price (in Lakhs)')

# Annotate the bars with average values
for index, value in enumerate(df_cleaned.groupby('fuel')['selling_price_lakhs'].mean().values):
    plt.text(index, value, f'{value:.2f}', ha='center', va='bottom')

plt.show()

# Average Selling Price by Seller Type
plt.figure(figsize=(10, 6))
sns.barplot(data=df_cleaned, x='seller_type', y='selling_price_lakhs', estimator=np.mean, palette='Set1')
plt.title('Average Selling Price by Seller Type')
plt.xlabel('Seller Type')
plt.ylabel('Average Selling Price (in Lakhs)')

# Annotate the bars with average values
for index, value in enumerate(df_cleaned.groupby('seller_type')['selling_price_lakhs'].mean().values):
    plt.text(index, value, f'{value:.2f}', ha='center', va='bottom')

plt.show()

# Create bins for Engine Size
engine_bins = [0, 1000, 1500, 2000, 2500, 3000, df_cleaned['engine'].max()]
engine_labels = ['<1000cc', '1000-1500cc', '1500-2000cc', '2000-2500cc', '2500-3000cc', '>3000cc']
df_cleaned['engine_bin'] = pd.cut(df_cleaned['engine'], bins=engine_bins, labels=engine_labels)

# Create bins for Mileage
mileage_bins = [0, 10, 15, 20, 25, 30, df_cleaned['mileage(km/ltr/kg)'].max()]
mileage_labels = ['<10 kmpl', '10-15 kmpl', '15-20 kmpl', '20-25 kmpl', '25-30 kmpl', '>30 kmpl']
df_cleaned['mileage_bin'] = pd.cut(df_cleaned['mileage(km/ltr/kg)'], bins=mileage_bins, labels=mileage_labels)

# Average Selling Price by Engine Size Bin
plt.figure(figsize=(12, 6))
sns.barplot(data=df_cleaned, x='engine_bin', y='selling_price_lakhs', estimator=np.mean, palette='Set3')
plt.title('Average Selling Price by Engine Size Bin')
plt.xlabel('Engine Size (cc)')
plt.ylabel('Average Selling Price (in Lakhs)')

# Annotate the bars with average values
for index, value in enumerate(df_cleaned.groupby('engine_bin')['selling_price_lakhs'].mean().values):
    plt.text(index, value, f'{value:.2f}', ha='center', va='bottom')

plt.xticks(rotation=45)
plt.show()

# Average Selling Price by Mileage Bin
plt.figure(figsize=(12, 6))
sns.barplot(data=df_cleaned, x='mileage_bin', y='selling_price_lakhs', estimator=np.mean, palette='Set2')
plt.title('Average Selling Price by Mileage Bin')
plt.xlabel('Mileage (kmpl)')
plt.ylabel('Average Selling Price (in Lakhs)')

# Annotate the bars with average values
for index, value in enumerate(df_cleaned.groupby('mileage_bin')['selling_price_lakhs'].mean().values):
    plt.text(index, value, f'{value:.2f}', ha='center', va='bottom')

plt.xticks(rotation=45)
plt.show()


# new feature car_age



# Get the current year
current_year = datetime.datetime.now().year

# Calculate car age
df_cleaned['car_age'] = current_year - df_cleaned['year']

# Display the first few rows to verify
print(df_cleaned[['year', 'car_age']].head())



# Identify categorical columns
cat_columns = df_cleaned.select_dtypes(include=['category']).columns

# Get unique values for each categorical column
unique_values_categorical = {col: df_cleaned[col].unique() for col in cat_columns}

# Display unique values for categorical columns
for col, values in unique_values_categorical.items():
    print(f"Unique values in '{col}': {values}")


# Initialize a dictionary to store label encoders
label_encoders = {}

# Perform label encoding and create new columns
for col in cat_columns:
    le = LabelEncoder()
    df_cleaned[f'{col}_encoded'] = le.fit_transform(df_cleaned[col])
    label_encoders[col] = le  # Store the encoder for future reference

# Display the transformed DataFrame
print(df_cleaned.head())  # Display the first few rows of the updated DataFrame

# Display the mapping of categories to numerical values
for col, le in label_encoders.items():
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"Mapping for '{col}': {mapping}")



# Save df_cleaned as a new CSV file named 'cleaned_car.csv'
#df_cleaned.to_csv('cleaned_car.csv', index=False)


#--------------------MACHINE LEARNING PART-----------------------

# Step 2: Load Data and Preprocess
# Load your cleaned dataset
df2 = pd.read_csv('cleaned_car.csv')

# Define features and target variable
X = df2.drop(columns=['selling_price', 'name', 'year', 'engine_bin', 'mileage_bin', 'fuel', 'transmission', 'seller_type', 'owner', 'selling_price_lakhs'])  # Drop unnecessary columns
y = df2['selling_price']

# Remove Rows with Missing Values
X = X.dropna()
y = y[X.index]  # Align the target variable with the features

# Identify numerical features to scale
numerical_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Scale numerical features
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

print("\nScaled Data for ML Model --")
print(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train Multiple Models
models = {
    "Linear Regression": LinearRegression(),
    "SVR": SVR(),
    "Decision Tree": DecisionTreeRegressor(),
    "KNN": KNeighborsRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "XGBoost": xgb.XGBRegressor()
}

# Step 4: Evaluate Models
results = {}
feature_importance_dict = {}  # Dictionary to store feature importance for each model
predictions = {}  # Dictionary to store predictions for XGBoost and Random Forest

for model_name, model in models.items():
    model.fit(X_train, y_train)  # Train the model
    y_pred = model.predict(X_test)  # Make predictions
    
    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Store results
    results[model_name] = {
        "Mean Absolute Error": mae,
        "Mean Squared Error": mse,
        "R² Score": r2
    }

    # Calculate and store feature importance for tree-based models
    if model_name in ["Random Forest", "XGBoost"]:
        importances = model.feature_importances_  # Get feature importances
        feature_importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        # Store feature importance in the dictionary
        feature_importance_dict[model_name] = feature_importance_df
        
        # Store predictions for actual vs. predicted comparison
        predictions[model_name] = y_pred

        # Print feature importance
        print(f"--- {model_name} Feature Importance ---")
        print(feature_importance_df)
        print("\n")

# Print the evaluation results
for model_name, metrics in results.items():
    print(f"--- {model_name} Evaluation Results ---")
    print(f"Mean Absolute Error (MAE): {metrics['Mean Absolute Error']:.4f}")
    print(f"Mean Squared Error (MSE): {metrics['Mean Squared Error']:.4f}")
    print(f"R² Score: {metrics['R² Score']:.4f}\n")

# Optional: Plot feature importance for Random Forest and XGBoost
for model_name in ["Random Forest", "XGBoost"]:
    if model_name in feature_importance_dict:
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_dict[model_name]['Feature'], feature_importance_dict[model_name]['Importance'])
        plt.xlabel("Feature Importance")
        plt.title(f"{model_name} Feature Importance")
        plt.show()

# Step 5: Compare Actual vs. Predicted Values for XGBoost and Random Forest
for model_name in ["Random Forest", "XGBoost"]:
    if model_name in predictions:
        # Create a DataFrame with actual and predicted values
        comparison_df = pd.DataFrame({
            "Actual": y_test,
            "Predicted": predictions[model_name]
        })
        
        # Calculate the difference
        comparison_df['Difference'] = comparison_df['Predicted'] - comparison_df['Actual']
        
        # Format the DataFrame to avoid scientific notation
        comparison_df["Actual"] = comparison_df["Actual"].apply(lambda x: f"{x:,.2f}")
        comparison_df["Predicted"] = comparison_df["Predicted"].apply(lambda x: f"{x:,.2f}")
        comparison_df["Difference"] = comparison_df["Difference"].apply(lambda x: f"{x:,.2f}")
        
        # Display the last 10 actual vs predicted values
        print(f"--- {model_name} Actual vs. Predicted Values ---")
        print(comparison_df.tail(10).to_string(index=True))  # Display formatted values
        print("\n")

# XGBoost Regressor and RandomForest Regressor performed Best among all models

