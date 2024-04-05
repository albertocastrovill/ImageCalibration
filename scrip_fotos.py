import cv2
import time
import os

# Crear la carpeta para las imágenes si no existe
folder_name = 'cheess_images'
os.makedirs(folder_name, exist_ok=True)

# Inicializar la cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se puede abrir la cámara")
    exit()

i=10
# Tomar 60 fotos, una cada 2 segundos
for i in range(40):
    # Capturar un frame
    ret, frame = cap.read()

    if not ret:
        print("No se pudo leer el frame. Saliendo...")
        break

    # Mostrar el frame en una ventana para poder posicionar el tablero
    cv2.imshow('Camera', frame)

    # Esperar un poco antes de la próxima captura para que se pueda ver en pantalla
    key = cv2.waitKey(2000)  # Espera 1000 ms

    # Guardar el frame como imagen
    filename = os.path.join(folder_name, f'foto_{i+1}.png')
    cv2.imwrite(filename, frame)
    print(f'Imagen {i+1} guardada en {filename}')

    # Esperar el tiempo restante para completar los 2 segundos
    time.sleep(2)  # Ya esperamos 0.5 segundos con cv2.waitKey

    if key & 0xFF == ord('q'):  # Permitir salir con 'q'
        break

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()

