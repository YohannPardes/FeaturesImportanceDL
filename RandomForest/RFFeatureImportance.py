import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# Load the CSV file (Upload your file)
train_csv = "../train_30h.csv"  # Replace with your actual file path
df = pd.read_csv(train_csv)

# Extract features (from the third column to the second last column)
X = df.iloc[:, 2:-1]  # Adjust indexing based on your file structure
y = df.iloc[:, -1]  # Last column as labels

test_csv = "../test_30h.csv"  # Replace with your actual file path
df = pd.read_csv(test_csv)

X_test = df.iloc[:, 2:-1]
y_test = df.iloc[:, -1]

# Train the Random Forest model
rf = RandomForestClassifier(n_estimators=2000, random_state=50, n_jobs=-1)
rf.fit(X, y)

# Compute Gini Importance (MDI)
feature_importances = rf.feature_importances_
mdi_df = pd.DataFrame({"Feature": X.columns, "Importance": feature_importances})
mdi_df = mdi_df.sort_values(by="Importance", ascending=False)

# Compute Permutation Importance (MDA)
perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=50, scoring="accuracy")
mda_df = pd.DataFrame({"Feature": X.columns, "Importance": perm_importance.importances_mean})
mda_df = mda_df.sort_values(by="Importance", ascending=False)

# Display results
print(mdi_df.to_string())  # Display in console
print()
print(mda_df.to_string())
