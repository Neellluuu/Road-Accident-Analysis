import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("data/accident_data.csv")

# Encode categorical columns
label_encoders = {}
for column in ["weather", "road_condition", "seatbelt", "severity"]:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Features and target
X = data.drop("severity", axis=1)
y = data["severity"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Custom Prediction
sample_accident = [[90, 
                    label_encoders["weather"].transform(["Rain"])[0],
                    label_encoders["road_condition"].transform(["Wet"])[0],
                    2,
                    label_encoders["seatbelt"].transform(["No"])[0]]]

predicted_severity = model.predict(sample_accident)
severity_label = label_encoders["severity"].inverse_transform(predicted_severity)

print("Predicted Accident Severity:", severity_label[0])
