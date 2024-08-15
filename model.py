import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
import time

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris_data = load_iris(as_frame=True)
x = iris_data.data
y = iris_data.target
x_train,x_valid,y_train,y_valid = train_test_split(x,y,test_size=0.8, train_size=0.2,random_state=42)


model = keras.Sequential([
    layers.Input(shape=[4]),
    layers.Dense(24, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(24, activation='relu'),
    layers.Dense(3,activation='softmax')
])
 
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(
    x_train, y_train,
    validation_data=(x_valid, y_valid),
    batch_size=50,
    epochs=300,
    verbose=0,
)

# Save the TensorFlow model
tf_model_path = "./Model1.keras" 
model.save(tf_model_path)

model = load_model("./Model1.keras")
model.summary()
start = time.process_time()
y_preds = model.predict(x_valid)
y_preds = np.argmax(y_preds, axis=1)
print('Prediction time:',time.process_time()-start)    

print("Accuracy: ",accuracy_score(y_valid,y_preds))   

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
plt.show()