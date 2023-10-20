import tensorflow as tf
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder

def initialize_model(model_name = 'forest_acoustic_monitoring_model.keras'):
    y = np.load('y_class.npy')
    label_encoder = LabelEncoder()
    label_encoder.fit_transform(y)
    model = tf.keras.models.load_model(model_name)
    return model, label_encoder

def model_summary(model):
    return model.summary()

def predict_class(filename, model, label_encoder):
    audio, sample_rate = librosa.load(filename)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    mfccs_scaled_features=mfccs_scaled_features.reshape(1, -1)
    x_predict = model.predict(mfccs_scaled_features)
    predicted_label = np.argmax(x_predict, axis=1)
    prediction_class = label_encoder.inverse_transform(predicted_label)
    return prediction_class

if __name__ == '__main__':
    model, label_encoder = initialize_model()
    filepath = ''
    print(predict_class(filepath, model, label_encoder))
