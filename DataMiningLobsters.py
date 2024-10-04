import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, r2_score

# Load the dataset in Python using read_excel() from pandas
lobster_df = pd.read_excel("/Users/ibensusan3/Desktop/Python/Project_DataMining/Lobster_Data_Set_-_[2925].xlsx")

# Display the first preview the dataframe to understand its structure
print(lobster_df)

# Assumed that the values indicated as mm and the values indicated as g are wrong, collumns chose to convertion
size_conversion = ['Length(mm)', 'Diameter(mm)', 'Height(mm)']
weight_conversion = ['WholeWeight(g)','ShuckedWeight(g)','SellWeight(g)']

# Convertion factors 
conversion_factor_size = 100
conversion_factor_weight = 10

# For loop to iterate the size convertion
for column in size_conversion:
    lobster_df[column] = lobster_df[column] * conversion_factor_size

# For loop to iterate the weight convertion
for column in weight_conversion:
    lobster_df[column] = lobster_df[column] * conversion_factor_weight

# Checking that the convertion
print(lobster_df.head(5))

# Searching and counting missing values
missing_values = lobster_df.isnull().sum() 

# Counting how many lobster we have per 'Sex'
sex_distribution = lobster_df['Sex'].value_counts()

# Counting 0 values in the dataframe, assuming that the only column allow to have them is 'Spots' column
zero_values_counts = (lobster_df == 0).sum()

print(missing_values, sex_distribution, zero_values_counts)

# Filter the lobster_df to include only rows where the 'Sex' column is 'M', 'F', or 'I'
lobster_df = lobster_df[lobster_df['Sex'].isin(['M', 'F', 'I'])]

# Replace zero values with NaN in all columns except 'Spots'
lobster_df= lobster_df.replace(0, np.nan)

# Calculate the median of each numerical column in lobster_df and store the results
medians = lobster_df.median(numeric_only=True)

# Fill missing values in lobster_df with their corresponding median values
lobster_df.fillna(medians, inplace=True)

# Calculate the total number of missing (NaN) values in each column of lobster_df after imputation
missing_values_after = lobster_df.isna().sum()
# Calculate and store the count of zero values in each column of lobster_df after imputation
zero_values_after = (lobster_df == 0).sum()

print(missing_values_after, zero_values_after)

# Generate descriptive statistics for numerical columns in the lobster_df DataFrame
print(lobster_df.describe())

# Set the seaborn plot background to "whitegrid" theme for better visibility and contrast
sns.set_theme(style="whitegrid")

# Define a list of column names related to the size measurements of lobsters
size_columns = ['Length(mm)', 'Diameter(mm)', 'Height(mm)']
# Define a list of column names related to the weight measurements of lobsters
weight_columns = ['WholeWeight(g)', 'ShuckedWeight(g)', 'SellWeight(g)']

# Create a new Matplotlib figure with a width of 18 inches and a height of 6 inches
plt.figure(figsize=(18, 6)) 

# Loop through each size-related column and create a subplot for a boxplot of that column's data grouped by 'Sex'
for i, column in enumerate(size_columns):
    plt.subplot(1, 3, i+1) # Positioning each subplot in a 1 row by 3 columns grid
    sns.boxplot(x='Sex', y=column, data=lobster_df) 
    plt.title(f'Distribution of {column}') # Title for each subplot
    plt.xlabel('Sex') # X-axis label
    plt.ylabel('mm') # Y-axis label

# Title for the figure
plt.suptitle('Distribution of size variables by sex', fontsize=16, y=1.05)
# Adjust the layout
plt.tight_layout()
# Display the figure
plt.show()


plt.figure(figsize=(18, 6)) 

# Loop through each weight-related column and create a subplot for a boxplot of that column's data grouped by 'Sex'
for i, column in enumerate(weight_columns):
    plt.subplot(1, 3, i+1)
    sns.boxplot(x='Sex', y=column, data=lobster_df)
    plt.title(f'Distribution of {column}')
    plt.xlabel('Sex')
    plt.ylabel('g')

plt.suptitle('Distribution of weight variables by sex', fontsize=16, y=1.05)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6)) 

sns.boxplot(x='Sex', y='Spots', data=lobster_df)
plt.title('Distribution Spots by Sex')
plt.xlabel('Sex')
plt.ylabel('Spots')
plt.tight_layout()
plt.show()

numeric_columns = ['Length(mm)', 'Diameter(mm)', 'Height(mm)', 'WholeWeight(g)', 'ShuckedWeight(g)', 'SellWeight(g)','Spots']
# Create a new DataFrame, lobster_df_numeric, containing only the specified numerical columns from lobster_df
lobster_df_numeric = lobster_df[numeric_columns]

# Calculate the first quartile (25th percentile) for each numerical column
Q1 = lobster_df_numeric.quantile(0.25)
# Calculate the third quartile (75th percentile) for each numerical column
Q3 = lobster_df_numeric.quantile(0.75)
# Compute the Interquartile Range (IQR) by subtracting Q1 from Q3 for each column
IQR = Q3 - Q1

# Calculate the lower bound for outlier detection in each numerical column
lower_bound = Q1 - 1.5 * IQR
# Calculate the upper bound for outlier detection in each numerical column
upper_bound = Q3 + 1.5 * IQR

# Calculate the total count of outliers for each numeric column in lobster_df_numeric
outliers = ((lobster_df_numeric < lower_bound) | (lobster_df_numeric > upper_bound)).sum()

#print(outliers)

# Create a boolean df indicating whether each numeric value in lobster_df_numeric is not an outlier
not_outlier = (lobster_df_numeric > lower_bound) & (lobster_df_numeric < upper_bound)

# Generate a boolean Series indicating rows without any outlier values across all numeric columns
rows_without_any_outlier = not_outlier.all(axis=1)

# Create a new DataFrame from lobster_df that includes only rows without any outliers
lobster_df_free_outliers = lobster_df[rows_without_any_outlier]

# Print the count of outliers for each numerical column in lobster_df_numeric
print(lobster_df_free_outliers)

# Set the seaborn plot background to "whitegrid" theme for better visibility and contrast
sns.set_theme(style="whitegrid")

# Create a new Matplotlib figure with a width of 18 inches and a height of 6 inches
plt.figure(figsize=(18, 6)) 

# Loop through each size-related column and create a subplot for a boxplot of that column's data grouped by 'Sex'
for i, column in enumerate(size_columns):
    plt.subplot(1, 3, i+1) # Positioning each subplot in a 1 row by 3 columns grid
    sns.boxplot(x='Sex', y=column, data=lobster_df_free_outliers) 
    plt.title(f'Distribution of {column} no outliers') # Title for each subplot
    plt.xlabel('Sex') # X-axis label
    plt.ylabel('mm') # Y-axis label

# Title for the figure
plt.suptitle('Distribution of size variables by sex', fontsize=16, y=1.05)
# Adjust the layout
plt.tight_layout()
# Display the figure
plt.show()


plt.figure(figsize=(18, 6)) 

# Loop through each weight-related column and create a subplot for a boxplot of that column's data grouped by 'Sex'
for i, column in enumerate(weight_columns):
    plt.subplot(1, 3, i+1)
    sns.boxplot(x='Sex', y=column, data=lobster_df_free_outliers)
    plt.title(f'Distribution of {column} no outliers')
    plt.xlabel('Sex')
    plt.ylabel('g')

plt.suptitle('Distribution of weight variables by sex', fontsize=16, y=1.05)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6)) 

sns.boxplot(x='Sex', y='Spots', data=lobster_df_free_outliers)
plt.title('Distribution Spots no outliers by Sex')
plt.xlabel('Sex')
plt.ylabel('Spots')
plt.tight_layout()
#plt.show()

# Select features of interest for the machine learning model 
X = lobster_df_free_outliers[['Length(mm)', 'WholeWeight(g)', 'Spots']] 
# Initialize the standard scaler
scaler = StandardScaler()
# Scale the selected features to have zero mean and unit variance
X_scaled = scaler.fit_transform(X)

# Initialize the KMeans clustering algorithm to partition the data into 3 clusters
kmeans = KMeans(n_clusters=3)  
# Fit the KMeans model to the scaled dataset
kmeans.fit(X_scaled)
# Retrieve the cluster labels assigned to each data point
clusters = kmeans.labels_

# Assign cluster labels to each row in lobster_df_free_outliers DataFrame
lobster_df_free_outliers['Cluster'] = clusters

# Initialize PCA with 2 components to reduce the dimensionality of the scaled data
pca = PCA(n_components=2)
# Fit the PCA model to the scaled data and transform it into 2 principal components
X_pca = pca.fit_transform(X_scaled)  

# Create a DataFrame from the PCA-transformed data with columns named 'PC1' and 'PC2'
pca_df = pd.DataFrame(data = X_pca, columns = ['PC1', 'PC2'])
# Assign cluster labels to each data point in the PCA DataFrame
pca_df['Cluster'] = clusters  

# Plot a scatter plot of the two principal components
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis', alpha=0.5)
plt.title('PCA Clusters')
plt.show()

# Loop through each unique cluster label to print descriptive statistics for that cluster
for i in set(clusters):
    cluster_data = lobster_df_free_outliers[lobster_df_free_outliers['Cluster'] == i]
    print(f"Cluster {i} Descriptive Statistics :")
    print(cluster_data[['Length(mm)', 'WholeWeight(g)', 'Spots']].describe(), "\n")
    
# Calculate the average silhouette score to evaluate the clustering quality
silhouette_avg = silhouette_score(X_scaled, clusters)
print(f"The silhouette coefficient average: {silhouette_avg}")

# Initialize a list to store the sum of squared distances for each k
sum_of_squared_distances = []
# Define the range of k values to test
K = range(1, 10)

# Loop over each value of k
for k in K:
    # Initialize a KMeans instance with k clusters
    km = KMeans(n_clusters=k)
    # Fit the KMeans model to the scaled data
    km = km.fit(X_scaled)
    # Append the model's inertia (sum of squared distances to nearest cluster center) to the list
    sum_of_squared_distances.append(km.inertia_)

# Plot the sum of squared distances for each k value using a line plot with 'x' markers
plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('Number of clusters k')
plt.ylabel('Sum of squares within the cluster')
plt.title('Elbow Method for Determining Optimal k')
plt.show()

# Convert the 'Sex' column in lobster_df_free_outliers into dummy variables, creating a separate column for each category
lobster_df_with_dummies = pd.get_dummies(lobster_df_free_outliers, columns=['Sex'], drop_first=False)

# Calculate the correlation matrix for all columns in lobster_df_with_dummies
correlation_matrix = lobster_df_with_dummies.corr()

# Generate a heatmap from the correlation matrix.
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Select the 'Length(mm)' column as the independent variable in a 2D format
X = lobster_df_free_outliers[['Length(mm)']]
# Select the 'WholeWeight(g)' column as the dependent variable
y = lobster_df_free_outliers['WholeWeight(g)']  

# Apply natural logarithm transformation to the independent variable 'Length(mm)'
X_log = np.log(X)
# Apply natural logarithm transformation to the dependent variable 'WholeWeight(g)'
y_log = np.log(y)

# Create an instance of the LinearRegression model
model = LinearRegression()
# Train the model using the logarithmically transformed independent and dependent variables
model.fit(X_log, y_log)

# Extract the slope coefficient of the linear regression model
b = model.coef_[0]
# Extract the intercept of the linear regression model in log-log space
log_a = model.intercept_

# Calculate the original scale intercept of the linear regression model by exponentiating log_a
a = np.exp(log_a)

print(f'a = {a}')
print(f'b = {b}')

#Define weigth_lenght_relationship() to calculate the estimated weight of a lobster based on its length.
def weigth_lenght_relationship(L):
    return 4.429049560521801e-05 * L ** 3.0250628550184806

## Generate values from 1 to 100
L_values = np.linspace(1,100)
# Calculate the corresponding weight for each length value using the relationship
W_values = weigth_lenght_relationship(L_values)

# Plot the weight-length relationship using L_values and W_values
plt.figure(figsize=(10, 6))
plt.plot(L_values, W_values, label='W = a* L^b')
plt.xlabel('Length')
plt.ylabel('Weigth')
plt.title('Visualisation Weigth-Length Relationship')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
# Plot the weight-length relationship as 'W = a * L^b'
plt.plot(L_values, W_values, label='W = a * L^b', color='blue')

# Overlay a scatter plot showing real, observed data points of lobster lengths and weights.
plt.scatter(lobster_df_free_outliers['Length(mm)'], lobster_df_free_outliers['WholeWeight(g)'], alpha=0.6, color='green', label='Real data')

# Add the labels and title only once, as it is now a combined chart.
plt.xlabel('Length(mm)')
plt.ylabel('WholeWeight(g)')
plt.title('Comparative Visualization of Weight-Length Relationship')
plt.legend()
plt.grid(True)
plt.show()

# Create a custom scoring object using the R-squared score to evaluate model performance
scorer = make_scorer(r2_score)

# Evaluate the model's performance using R-squared scores from cross-validation
scores = cross_val_score(model, X_log, y_log, cv=5, scoring=scorer)

# Print the mean R-squared value and the standard deviation 
print(f"R-squared: {scores.mean():.2f}, Standard deviation: {scores.std():.2f}")