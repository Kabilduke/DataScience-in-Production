import numpy as np
import pandas as pd

from keras import Input, layers, models, regularizers
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from loguru import logger

from build.pipeline.Data_pre import preprocess_data
from config.config import settings


def build_model():
    logger.info('Building and training the model...')
    X_train, X_test, y_train, y_test = preprocess_data()

    class_weights = weight_balancing(y_train)
    model = network(X_train, y_train, class_weights)
    predictiction = predict(model, X_test)

    classy_report = report(y_test, predictiction)
    logger.info('Classification Report:\n', classy_report)
    loss, accuracy = evaluate(model, X_test, y_test)
    logger.info(f'Model training completed - Loss: {loss:2f} - Accuracy: {accuracy:2f}')
    
    save_model(model)

#Weight balancing
def weight_balancing(y_train: pd.Series) -> dict:
    weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(weights))
    return class_weights

#Model Training (Sequential API)
def network(X_train, y_train, class_weights) -> models.Sequential:
    NNmodel = models.Sequential([
        Input(shape = (X_train.shape[1], ), name = 'input_layer'),
        layers.Dense(64, activation='elu', kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    NNmodel.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    NNmodel.fit(
        X_train,
        y_train,
        class_weight= class_weights,
        epochs=25,
        batch_size=32, 
        validation_split=0.2
    )
    return NNmodel


def predict(NNmodel, X_test):
    y_prob_pred = NNmodel.predict(X_test)
    y_pred = (y_prob_pred > 0.5).astype(int).flatten()
    return y_pred

def report(y_test, y_pred):
    classy_report = classification_report(y_test, y_pred)
    return classy_report

def evaluate(NNmodel, X_test, y_test) -> tuple:
    loss, accuracy = NNmodel.evaluate(X_test, y_test)
    return loss, accuracy
    

def save_model(NNmodel: models.Sequential) -> None:
    logger.info(f'Saving model to {settings.model_path}/{settings.model_name}')
    NNmodel.save(f'{settings.model_path}/{settings.model_name}')  # model_path
