import os
import tensorflow as tf
import pathlib

# Definir la ruta a los datos
data_dir = pathlib.Path('../datos')  # Ruta absoluta o relativa a tus datos

# Verificar que la ruta es correcta
if not data_dir.exists():
    raise FileNotFoundError(f"No existe el directorio: {data_dir}")

# Ajustar el tamaño de la imagen según tus necesidades
img_height = 224
img_width = 224

# Función para procesar la ruta de la imagen y su etiqueta
def process_path(file_path):
    label = tf.strings.split(file_path, os.path.sep)[-2]
    label = tf.where(label == 'plastic', 0, tf.where(label == 'paper', 1, 2))
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_height, img_width])
    img = img / 255.0
    return img, label

# Cargar y preprocesar las imágenes de entrenamiento y validación
train_ds = tf.data.Dataset.list_files(str(data_dir / 'train' / '*/*'), shuffle=True)
val_ds = tf.data.Dataset.list_files(str(data_dir / 'val' / '*/*'), shuffle=True)

train_ds = train_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Definir el tamaño del lote (batch size)
batch_size = 32

train_ds = train_ds.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_ds = val_ds.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
