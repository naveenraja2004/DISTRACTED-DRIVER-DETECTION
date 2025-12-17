import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models

# List of CSV files for all classes
csv_files = [
    "facial_landmarks_Drowsy.csv",
    "facial_landmarks_Safe.csv",
    "facial_landmarks_Yawn.csv",
    "facial_landmarks_Face_tilt.csv"
]

# Load and concatenate all CSVs
dfs = [pd.read_csv(f) for f in csv_files]
df = pd.concat(dfs, ignore_index=True)

# Ensure the last column is 'class'
df.rename(columns={df.columns[-1]: "class"}, inplace=True)

# Separate features and labels
X = df.iloc[:, :-1].values  # landmark features
y = df["class"].values      # class labels

# Encode labels to integers
le = LabelEncoder()
y = le.fit_transform(y)

# Train-test split (stratified to maintain class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build the neural network
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(len(le.classes_), activation='softmax')  # Output layer
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train the model
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32
)

# Evaluate on test data
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc*100:.2f}%")

# Save the trained model
model.save("driver_behavior_model.h5")
print("Model saved as driver_behavior_model.h5")
