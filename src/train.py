import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import os
from data import get_sp
from models import getModel

x_train, y_train, x_val, y_val, x_test, y_test = get_sp()

model = getModel()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=[tf.keras.metrics.AUC(), 'accuracy'])

logdir = os.path.join("../logs")
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10),
    ModelCheckpoint(filepath=os.path.join(logdir, 'best_model.h5'), monitor='val_loss',
                    save_best_only=True),
]

history = model.fit(x_train,
                    y_train,
                    epochs=1000,
                    callbacks=callbacks,
                    verbose=2,
                    batch_size=32,
                    validation_data=(x_val, y_val))

model.evaluate(x_test, y_test, verbose=2)

predictions = model.predict(x_test)
output_list = [0, 0]
label_list = [0, 0]

for i in range(predictions.shape[0]):
    output_list[round(predictions[i][0])] += 1
    label_list[round(y_test[i])] += 1


print("predictions: ", output_list)
print("labels: ", label_list)
