import time
import random
import qrcode
import cv2
import numpy as np
import tensorflow as tf
import serial
from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306

# Configuración del modelo de TensorFlow
model = tf.keras.models.load_model('waste_classifier_model.keras')  # Ajusta esta ruta

# Inicialización de la comunicación serial con Arduino
arduino = serial.Serial('COM3', 9600)  # Ajusta el puerto COM según tu configuración

# Dirección I2C de la pantalla OLED (No es necesaria en Windows)
# I2C_ADDR = 0x3C

# Dimensiones de la pantalla OLED
WIDTH = 128
HEIGHT = 64

# Inicialización del bus I2C para la pantalla OLED (omitido en Windows)
# serial_interface = i2c(port=1, address=I2C_ADDR)
# oled = ssd1306(serial_interface)

# Función para mover el servomotor y mostrar QR en la pantalla OLED (adaptada)
def mover_servomotor_y_mostrar_qr(servomotor):
    arduino.write(f'MOVER:{servomotor}\n'.encode())  # Enviar comando para mover el servomotor
    time.sleep(1)  # Esperar a que se complete el movimiento

    # Generar código QR aleatorio
    codigo_qr = f"DESCUENTO-{random.randint(1000, 9999)}"

    # Mostrar código QR en la consola como ejemplo en lugar de en la OLED
    print(f"Código QR: {codigo_qr}")

# Función para detectar objetos usando TensorFlow (mantenida)
def detectar_objeto(frame):
    img = cv2.resize(frame, (224, 224))  # Ajusta el tamaño según tu modelo
    img = np.expand_dims(img, axis=0) / 255.0
    pred = model.predict(img)
    objeto_detectado = np.argmax(pred)
    return objeto_detectado  # Ajusta según las clases de tu modelo

# Bucle principal (mantenido)
def main():
    cap = cv2.VideoCapture(0)  # Inicializar captura de video

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            objeto_detectado = detectar_objeto(frame)
            if objeto_detectado == 0:  # Papel
                mover_servomotor_y_mostrar_qr(1)  # Mover servomotor 1 y mostrar QR
                time.sleep(5)  # Esperar antes de la siguiente detección
            elif objeto_detectado == 1:  # Plástico
                mover_servomotor_y_mostrar_qr(2)  # Mover servomotor 2 y mostrar QR
                time.sleep(5)  # Esperar antes de la siguiente detección
            elif objeto_detectado == 2:  # Metal
                mover_servomotor_y_mostrar_qr(3)  # Mover servomotor 3 y mostrar QR
                time.sleep(5)  # Esperar antes de la siguiente detección

            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nSaliendo del programa...\n")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        arduino.close()
        # oled.cleanup() # Omitido para Windows

if __name__ == "__main__":
    main()








