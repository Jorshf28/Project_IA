import tensorflow as tf
import pathlib

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from src.Preproces_Datos import train_ds, val_ds, img_height, img_width, process_path, batch_size

# Definición del modelo usando la API funcional
inputs = Input(shape=(img_height, img_width, 3))
x = Conv2D(32, (3, 3), activation='relu')(inputs)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(3, activation='softmax')(x)  # Supongamos 3 clases: plástico, papel, metal

model = Model(inputs, outputs)

# Compilación del modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenamiento del modelo
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25
)

# Guardar el modelo entrenado
model.save('waste_classifier_model.keras')

# Cargar el modelo guardado
model = tf.keras.models.load_model('waste_classifier_model.keras')

# Evaluación del modelo
test_dir = pathlib.Path('../datos/test')
test_ds = tf.data.Dataset.list_files(str(test_dir/'*/*'), shuffle=False)

test_ds = test_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

test_loss, test_acc = model.evaluate(test_ds)
print('Precisión de las pruebas:', test_acc)

