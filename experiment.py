import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime

# Read CSV
data = pd.read_csv("Churn_Modelling.csv")
print(f"==>> data.head(): {data.head()}")

# Drop Not useful columns
data = data.drop(["RowNumber", "CustomerId", "Surname"], axis=1)
print(f"==>> data.head(): {data.head()}")

# Converts "Male" → 1, "Female" → 0.
label_encoder_gender = LabelEncoder()
data["Gender"] = label_encoder_gender.fit_transform(data["Gender"])
print(f"==>> data.head(): {data.head()}")

# - Converts it into 3 separate columns:
#     - `"Geography_France"`
#     - `"Geography_Germany"`
#     - `"Geography_Spain"`
onehot_encoder_geo = OneHotEncoder()
geo_encoder = onehot_encoder_geo.fit_transform(data[["Geography"]])
onehot_encoder_geo.get_feature_names_out(["Geography"])
geo_encoder_df = pd.DataFrame(
    geo_encoder.toarray(),
    columns=onehot_encoder_geo.get_feature_names_out(["Geography"]),
)
print(f"==>> geo_encoder_df: {geo_encoder_df}")

# Drops "Geography".
# Merge newly encoded columns
data = pd.concat([data.drop("Geography", axis=1), geo_encoder_df], axis=1)
print(f"==>> data.head(): {data.head()}")

# Saves label_encoder_gender and onehot_encoder_geo to reuse later.
with open("label_encoder_gender.pkl", "wb") as file:
    pickle.dump(label_encoder_gender, file)

with open("onehot_encoder_geo.pkl", "wb") as file:
    pickle.dump(onehot_encoder_geo, file)

# X = Features, contains all column except of the answer we want
# Y = Target, contains the just answer which is the column Exited. (churn label: 0 = stayed, 1 = left).
X = data.drop("Exited", axis=1)
y = data["Exited"]

# Splits 80% training data and 20% testing data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardizes X_train and X_test (mean = 0, std = 1).
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

# Define a Neural Network Model
# - **Input Layer**: `Dense(64, activation="relu")`
# - **Hidden Layer**: `Dense(32, activation="relu")`
# - **Output Layer**: `Dense(1, activation="sigmoid")`
#     - Uses `sigmoid` activation for binary classification.
model = Sequential(
    [
        Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ]
)

print(f"==>> model.summary(): {model.summary()}")

#  Compile the Model
# - **Optimizer**: `Adam` (adaptive learning).
# - **Loss Function**: `binary_crossentropy` (for binary classification).
# - **Metrics**: `"accuracy"`.
opt = tensorflow.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
print(f"==>> model.summary(): {model.summary()}")

# Define Callbacks
# - TensorBoard: Logs training progress.
# - EarlyStopping: Stops training when `val_loss` stops improving.
log_dir = "logs/fit"
tensorflow_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

early_stopping_callback = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

# Train the Model
# - Trains for 100 epochs (or stops early if validation loss doesn’t improve).
# - Uses training (`X_train`, `y_train`) and validation (`X_test`, `y_test`) data.
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    callbacks=[early_stopping_callback, tensorflow_callback],
)

# Save the Model
# Saves the trained model to a file (model.h5) for later use to predict real world problem
model.save("model.h5")
