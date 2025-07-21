import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from loguru import logger

from build.pipeline.Data import load_db


#Data Cleaning
def preprocess_data():
    logger.info('Preprocessing data...')
    df = load_db()

    df = encode(df)
    df = clean(df)
    X_train, X_test, y_train, y_test = data_splitting(df)
    X_train, X_test = stand(X_train, X_test)

    return X_train, X_test, y_train, y_test

def encode(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df['Gender'] == 'f', 'Gender'] = 'F'
    df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.drop(['Column1', 'Gender'], axis = 1)
    return df_clean

#Data Preprocessing
def data_splitting(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    x_label = df[['Age', 'BMI', 'Chol', 'TG', 'HDL', 'LDL', 'Cr', 'BUN']]
    y_label = df['Diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(
        x_label, 
        y_label, 
        test_size=0.2, 
        random_state=42, 
        stratify= y_label
    )
    return X_train, X_test, y_train, y_test

def stand(X_train, X_test) -> tuple[np.ndarray, np.ndarray]:
    Scaler = StandardScaler()
    X_train = Scaler.fit_transform(X_train)
    X_test = Scaler.transform(X_test)
    return X_train, X_test
