import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset into a Pandas DataFrame
df = pd.read_csv('data/hour.csv')

# Check the first 5 rows of the DataFrame
print(df.head())

# Check the data types of each column
print(df.dtypes)

print(df.describe())

# Check the missing values in the DataFrame
print(df.isnull().sum())

# Check the unique values in each column
print(df['count'].unique())

print(df['count'].mean())
print(df['count'].median())
print(df['count'].std())
print(df['registered'].min())
print(df['registered'].max())

# Check the distribution of a numeric variable
plt.hist(df['count'])
plt.show()

# Check the correlation between two numeric variables
corr = df['count'].corr(df['registered'])
print('Correlation:', corr)

# Check the relationship between a categorical variable and a numeric variable
sns.boxplot(x='categorical_variable', y='numeric_variable', data=df)
plt.show()

# Check the distribution of a datetime variable
df['datetime_variable'].dt.strftime('%Y-%m-%d').value_counts().plot(kind='bar')
plt.show()

# Check the relationship between two categorical variables
sns.countplot(x='categorical_variable1', hue='categorical_variable2', data=df)
plt.show()
