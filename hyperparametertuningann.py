import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from scikeras.wrappers import KerasClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# Read CSV
data = pd.read_csv("Churn_Modelling.csv")
# Drop Not useful columns
data = data.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

# Converts "Male" → 1, "Female" → 0.
label_encoder_gender = LabelEncoder()
data["Gender"] = label_encoder_gender.fit_transform(data["Gender"])
print(f"==>> data.head(): {data.head()}")

# - Converts it into 3 separate columns:
#     - `"Geography_France"`
#     - `"Geography_Germany"`
#     - `"Geography_Spain"`
onehot_encoder_geo = OneHotEncoder(handle_unknown="ignore")
geo_encoded = onehot_encoder_geo.fit_transform(data[["Geography"]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
)
print(f"==>> geo_encoder_df: {geo_encoded_df}")

# Drops "Geography".
# Merge newly encoded columns
data = pd.concat([data.drop("Geography", axis=1), geo_encoded_df], axis=1)

# Split the data into features and target
X = data.drop("Exited", axis=1)
y = data["Exited"]

# Splits 80% training data and 20% testing data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardizes X_train and X_test (mean = 0, std = 1).
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Save encoders and scaler for later use
with open("label_encoder_gender.pkl", "wb") as file:
    pickle.dump(label_encoder_gender, file)

with open("onehot_encoder_geo.pkl", "wb") as file:
    pickle.dump(onehot_encoder_geo, file)

with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)


# Define a common function
def create_model(neurons=32, layers=1):
    model = Sequential()
    model.add(Dense(neurons, activation="relu", input_shape=(X_train.shape[1],)))

    for _ in range(layers - 1):
        model.add(Dense(neurons, activation="relu"))

    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model


# Init a model with 1 layer and 32 neurons
model = KerasClassifier(layers=1, neurons=32, build_fn=create_model, verbose=1)

# Define the parameters of the model
param_grid = {"neurons": [16, 32, 64, 128], "layers": [1, 2], "epochs": [50, 100]}

# Using GridSearchCV, train the combination 4 * 2 * 2 = 16times
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, verbose=1)
grid_result = grid.fit(X_train, y_train)

# Loop through all tested parameter combinations and their corresponding mean accuracy
for mean_score, params in zip(
    grid_result.cv_results_["mean_test_score"], grid_result.cv_results_["params"]
):
    print(f"Mean Accuracy: {mean_score:.6f} using {params}")

# Print the best score of the parameter
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
