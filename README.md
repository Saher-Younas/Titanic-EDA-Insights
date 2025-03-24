
# Titanic Dataset - Exploratory Data Analysis (EDA)

## Overview
This project performs Exploratory Data Analysis (EDA) on the Titanic dataset. The analysis covers data cleaning, univariate and bivariate analysis, visualizations, and key insights to understand passenger survival factors.

## Dataset
The dataset used for this analysis is the Titanic dataset, which contains information on passengers, including their age, gender, class, fare, and survival status. [https://github.com/Saher-Younas/Titanic-EDA-Insights/blob/main/titanic.csv]

---

## Steps Performed
### 1. Importing Required Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

### 2. Loading the Dataset
```python
df = pd.read_csv('titanic.csv')
df.shape  # Check dataset dimensions
```

### 3. Checking for Missing Values
```python
df.isnull().sum()  # Count missing values in each column
```

### 4. Checking for Duplicates
```python
df.duplicated().sum()
```

### 5. Data Cleaning
#### Handling Missing Values
- Filled missing values in columns with a few missing values:
```python
df['age'].fillna(df['age'].median(), inplace=True)
df['fare'].fillna(df['fare'].mean(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
df['home.dest'].fillna('Unknown', inplace=True)
```
- Dropped columns with excessive missing values:
```python
df2 = df.drop(columns=['cabin', 'body', 'boat'], errors='ignore')
df2.isnull().sum()  # Verify missing values are handled
```

---

## Univariate Analysis
### 1. Survival Count
```python
plt.figure(figsize=(6, 4))
sns.countplot(x="survived", data=df2, palette="Blues")
plt.xticks([0, 1], ["Did Not Survive", "Survived"])
plt.title("Survival Count")
plt.show()
```
**Insight:** More passengers did not survive compared to those who did.

### 2. Age Distribution of Passengers
```python
plt.figure(figsize=(8, 5))
sns.histplot(df2['age'], bins=30, kde=True, color="skyblue")
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()
```
**Insight:** Most passengers were in their 20s and 30s, with a significant peak around 30.

### 3. Fare Distribution
```python
plt.figure(figsize=(8, 5))
sns.boxplot(x=df2['fare'], palette="Blues")
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.show()
```
**Insight:** Most passengers paid low fares, but there were some outliers who paid significantly higher fares.

### 4. Passenger Class Distribution
```python
plt.figure(figsize=(8, 5))
sns.countplot(x=df2['pclass'], palette="viridis")
plt.title("Passenger Class Distribution")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.show()
```
**Insight:** Most passengers were in 3rd class, followed by 1st and 2nd class.

### 5. Gender Distribution
```python
plt.figure(figsize=(8, 5))
sns.countplot(x=df2['sex'], palette="viridis")
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()
```
**Insight:** There were more male passengers than females on board.

---

## Bivariate Analysis
### 1. Age vs Fare Paid
```python
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df2['age'], y=df2['fare'], hue=df2['pclass'], palette="viridis")
plt.title("Age vs Fare Paid")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()
```
**Insight:** 1st-class passengers paid higher fares, while 3rd-class passengers paid the least.

### 2. Age Distribution by Survival
```python
plt.figure(figsize=(8, 5))
sns.violinplot(x=df2['survived'], y=df2['age'], palette="coolwarm")
plt.title("Age Distribution by Survival")
plt.xlabel("Survived")
plt.ylabel("Age")
plt.show()
```
**Insight:** Younger passengers had a higher survival rate than older ones.

### 3. Survival Rate by Passenger Class
```python
plt.figure(figsize=(8, 5))
sns.histplot(x=df2['pclass'], hue=df2['survived'], multiple='stack', palette="coolwarm", discrete=True)
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.xticks([1, 2, 3])
plt.show()
```
**Insight:** 1st-class passengers had the highest survival rate, while 3rd-class passengers had the lowest.

### 4. Survival Rate by Gender
```python
survived_count = df2.groupby('sex')['survived'].value_counts(normalize=True).unstack()
survived_count.plot(kind='bar', stacked=True, colormap='coolwarm', figsize=(8,5))
plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Proportion")
plt.legend(['Did Not Survive', 'Survived'], loc='upper right')
plt.show()
```
**Insight:** Females had a much higher survival rate than males.

### 5. Fare vs Survival
```python
plt.figure(figsize=(8, 5))
sns.boxplot(x=df2['survived'], y=df2['fare'], palette="coolwarm")
plt.title("Fare vs Survival")
plt.xlabel("Survival")
plt.ylabel("Fare")
plt.show()
```
**Insight:** Passengers who paid higher fares had a better chance of survival.

---

## Correlation Analysis
```python
plt.figure(figsize=(8, 6))
sns.heatmap(df2.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Titanic Dataset")
plt.show()
```
### Insights:
- Fare was positively correlated with survival.
- Passenger class (pclass) was negatively correlated with survival, meaning higher-class passengers had better survival chances.

---

## Conclusion
This Titanic EDA project revealed crucial insights into survival factors. Key findings include:
- More males than females were on board, but females had a higher survival rate.
- 1st-class passengers had the highest survival rate.
- Higher fares were associated with higher survival chances.
- Younger passengers had better survival rates than older passengers.

This analysis provides a clear understanding of the dataset and key survival factors on the Titanic.



