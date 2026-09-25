from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Features + target as a single DataFrame
df = housing.frame

# Quick check
print(df.head())
print(df.shape)

# Create the figs folder if it doesn't exist
os.makedirs("figs", exist_ok=True)

# Create a boxplot for median income
df.boxplot(column="MedInc")

plt.title("Boxplot of Median Income")
plt.ylabel("Median Income")

# Save the boxplot
plt.savefig("figs/boxplot.png", bbox_inches="tight")
plt.close()