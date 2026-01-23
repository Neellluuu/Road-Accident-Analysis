import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("../dataset/road_accidents.csv")

print("Dataset Preview:")
print(data.head())

# Total accidents by state
state_accidents = data.groupby("State").size()
print("\nAccidents by State:\n", state_accidents)

# Plot accidents by state
plt.figure(figsize=(8,5))
state_accidents.plot(kind="bar", color="orange")
plt.title("Road Accidents by State")
plt.xlabel("State")
plt.ylabel("Number of Accidents")
plt.show()

# Accident type analysis
plt.figure(figsize=(8,5))
sns.countplot(x="Accident_Type", data=data)
plt.title("Accidents by Type")
plt.xticks(rotation=30)
plt.show()
