import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("../dataset/it_support_tickets.csv")

print("First 5 Rows:")
print(df.head())

# Convert datetime columns
df['Created_Time'] = pd.to_datetime(df['Created_Time'])
df['Resolved_Time'] = pd.to_datetime(df['Resolved_Time'])

# Resolution time
df['Resolution_Hours'] = (
    df['Resolved_Time'] - df['Created_Time']
).dt.total_seconds() / 3600

# Created hour
df['Created_Hour'] = df['Created_Time'].dt.hour

# Department analysis
print("\nDepartment-wise Ticket Count:")
print(df['Department'].value_counts())

# Graph 1
df['Department'].value_counts().plot(
    kind='bar',
    figsize=(8,5),
    title="Department-wise Ticket Count"
)
plt.show()

# Graph 2
df['Created_Hour'].value_counts().sort_index().plot(
    kind='line',
    marker='o',
    figsize=(8,5),
    title="Peak Hours for Ticket Creation"
)
plt.show()

# Average resolution time
print("\nAverage Resolution Time:")
print(df['Resolution_Hours'].mean())

# Encoding
le = LabelEncoder()

for col in ['Department', 'Priority', 'Issue_Type',
            'Assigned_Team', 'Status', 'SLA_Breached']:
    df[col] = le.fit_transform(df[col])

# Features and target
X = df[['Department', 'Priority', 'Issue_Type',
        'Assigned_Team', 'Status',
        'Resolution_Hours', 'Created_Hour']]

y = df['SLA_Breached']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))