'''
Forest Acoustic Monitoring System (FAMSystem)
By Team - Gitgat
'''

import numpy as np
import librosa
import pandas as pd

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

metadata = pd.read_csv('metadata.csv')

def add_noise(wav):
  wav_n = wav + 0.009 * np.random.normal(0, 1, len(wav))
  return wav_n

def time_shift(wav, sr):
  wav_roll = np.roll(wav, int(sr/10))
  return wav_roll

def stretch_wav(wav):
  factor = 0.4
  wav_time_stch = librosa.effects.time_stretch(wav, rate=0.4)
  return wav_time_stch

def pitch_shift(wav, sr):
  wav_pitch_sf = librosa.effects.pitch_shift(y=wav, sr=sr, n_steps=-5)
  return wav_pitch_sf

def get_mfcc_features(audio, sample_rate):
  mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
  mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
  return mfccs_scaled_features

def extract_features(file_path):
  audio, sample_rate = librosa.load(file_path)
  augmented_audios = [audio,
                      add_noise(audio),
                      time_shift(audio, sample_rate),
                      stretch_wav(audio),
                      pitch_shift(audio, sample_rate)]
  augmented_features = []
  for a in augmented_audios:
    augmented_features.append(get_mfcc_features(a, sample_rate))

  return augmented_features

extracted_features = []
for index_num, row in metadata.iterrows():
  class_label = row['Class Name']
  file_name = row['Dataset File Name']
  file_path = f'/dataset/{file_name}'
  features = extract_features(file_path)
  for f in features:
    extracted_features.append([f, class_label])

features_df = pd.DataFrame(extracted_features, columns=['feature', 'class_name'])

#print(features_df.head(10))

X = np.array(features_df.feature.to_list())
y = np.array(features_df.class_name.to_list())
np.save('y_class.npy', y)

label_encoder = LabelEncoder()
y = to_categorical(label_encoder.fit_transform(y))

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=0)

num_labels=y.shape[1]
model=Sequential()

model.add(Dense(200,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(400))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(400))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

#print(model.summary())

num_epochs = 100
num_batch_size = 32
model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs,
          validation_data=(X_test, y_test), verbose=1)

model.save("forest_acoustic_monitoring_model.keras")
