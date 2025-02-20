#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#

import pandas as pd
import numpy as np
import os
import json
import gzip
import joblib
import zipfile

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def preprocess_data(zip_file_path):
    """ Limpieza de datos según las especificaciones. """
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        csv_filename = zip_ref.namelist()[0]
        with zip_ref.open(csv_filename) as f:
            df = pd.read_csv(f)

    # Crear la columna "Age" basada en el año de fabricación
    df["Age"] = 2021 - df["Year"]

    # Eliminar columnas "Year" y "Car_Name"
    df.drop(columns=["Year", "Car_Name"], inplace=True)

    return df


def split_data(df):
    """ Divide los datos en X (variables explicativas) e y (objetivo) """
    X = df.drop(columns=["Selling_Price"])
    y = df["Selling_Price"]
    return X, y


def build_pipeline():
    """ Crea un pipeline con OneHotEncoding, MinMaxScaler, SelectKBest y regresión lineal """

    categorical_features = ["Fuel_type", "Selling_Type", "Transmission"]

    preprocessor = ColumnTransformer([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ("scaler", MinMaxScaler(), slice(0, -1))  # Escalar todas las columnas excepto la última
    ], remainder="passthrough")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("select", SelectKBest(f_regression, k=5)),  # Selecciona las 5 características más relevantes
        ("regressor", LinearRegression())
    ])

    return pipeline


def optimize_hyperparameters(pipeline, X_train, y_train):
    """ Optimiza hiperparámetros usando GridSearchCV con validación cruzada """
    param_grid = {
        "select__k": [5, 10],  # Ajustar el número de características seleccionadas
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring="neg_mean_absolute_error", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"✅ Mejor error absoluto medio: {-grid_search.best_score_}")
    print(f"🔍 Mejores hiperparámetros: {grid_search.best_params_}")

    return grid_search  # Retorna el GridSearchCV completo


def save_model(model, file_path):
    """ Guarda el modelo optimizado en gzip """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with gzip.open(file_path, "wb") as f:
        joblib.dump(model, f)


def calculate_metrics(model, X, y, dataset_type):
    """ Calcula R2, error cuadrático medio (MSE) y error absoluto medio (MAD) """
    y_pred = model.predict(X)

    metrics = {
        "type": "metrics",
        "dataset": dataset_type,
        "r2": r2_score(y, y_pred),
        "mse": mean_squared_error(y, y_pred),
        "mad": mean_absolute_error(y, y_pred)
    }

    return metrics


def save_metrics(metrics_list, file_path):
    """ Guarda las métricas en un archivo JSON con el orden correcto """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        for metric in metrics_list:
            f.write(json.dumps(metric) + "\n")


def main():
    # Paso 1: Cargar y limpiar datos
    train_file = "files/input/train_data.csv.zip"
    test_file = "files/input/test_data.csv.zip"

    train_df = preprocess_data(train_file)
    test_df = preprocess_data(test_file)

    # Paso 2: Dividir datos en X e y
    X_train, y_train = split_data(train_df)
    X_test, y_test = split_data(test_df)

    # Paso 3: Construir pipeline
    pipeline = build_pipeline()

    # Paso 4: Optimizar hiperparámetros
    model = optimize_hyperparameters(pipeline, X_train, y_train)

    # Paso 5: Guardar modelo
    model_path = "files/models/model.pkl.gz"
    save_model(model, model_path)

    # Paso 6: Calcular métricas
    metrics_train = calculate_metrics(model.best_estimator_, X_train, y_train, "train")
    metrics_test = calculate_metrics(model.best_estimator_, X_test, y_test, "test")

    # Guardar métricas en JSON
    metrics_path = "files/output/metrics.json"
    save_metrics([metrics_train, metrics_test], metrics_path)

    print(f"✅ Modelo guardado en {model_path}. Métricas en {metrics_path}.")


if __name__ == "__main__":
    main()
