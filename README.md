# Project: Lobster Data Analysis and Modeling

This project involved analyzing a dataset of lobsters to derive insights about their physical attributes and predict certain characteristics using machine learning and statistical methods.

## Objectives:

1. **Data Preprocessing**:
    - Load and clean the dataset using pandas.
    - Convert specific units of measurement (mm to cm, grams to kilograms) to ensure consistency.
    - Handle missing data by imputing values with medians and handling zeros in non-appropriate columns.
    - Identify and remove outliers using the interquartile range (IQR) method.

2. **Descriptive Analysis**:
    - Explore the distribution of size and weight variables grouped by the sex of the lobsters.
    - Visualize the distribution of data using boxplots to compare the lobster sizes and weights across different sexes.
    - Analyze the number of outliers for each numerical variable and remove them for further analysis.

3. **Clustering**:
    - Use KMeans clustering to group lobsters based on physical features (length, weight, and spots).
    - Perform PCA to reduce the dimensions and visualize the clusters.
    - Calculate the silhouette score to evaluate the quality of clustering.

4. **Correlation Analysis**:
    - Use a correlation matrix to analyze relationships between various features in the dataset and visualize it using a heatmap.

5. **Weight-Length Relationship**:
    - Perform a regression analysis to understand the relationship between lobster length and weight.
    - Use logarithmic transformations to linearize the data.
    - Create a model to predict lobster weight based on its length, with the equation of the form: \( W = aL^b \).
    - Visualize the weight-length relationship and compare it to real observed data points.

6. **Model Evaluation**:
    - Evaluate the regression model using R-squared scores from cross-validation to ensure its accuracy and robustness.
    - Report the mean R-squared and standard deviation for the model performance.

## Tools and Libraries:

- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Matplotlib & Seaborn**: For data visualization.
- **Scikit-learn**: For clustering (KMeans), scaling, PCA, and regression analysis.

## Outcomes:

- A well-prepared and cleaned lobster dataset ready for analysis.
- Visualization and understanding of the distribution of physical attributes of lobsters by sex.
- Grouping of lobsters into clusters based on physical traits.
- A predictive model for estimating the weight of lobsters based on their length.
- Visualization of real vs predicted weight-length relationships and evaluation of the model performance using R-squared values.

## License

This project is licensed under the MIT License. 
