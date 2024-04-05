import cv2
import time
import os

# Crear la carpeta para las imágenes si no existe
folder_name = 'distorded_images'
os.makedirs(folder_name, exist_ok=True)

# Inicializar la cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se puede abrir la cámara")
    exit()

i = 0
start_time = time.time()  # Obtener el tiempo inicial

# Tomar fotos cada 5 segundos
while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer el frame. Saliendo...")
        break

    # Mostrar el frame en una ventana
    cv2.imshow('Camera', frame)

    if time.time() - start_time >= 5:  # Verificar si han pasado 5 segundos
        # Guardar el frame como imagen
        filename = os.path.join(folder_name, f'foto_{i+1}.png')
        cv2.imwrite(filename, frame)
        print(f'Imagen {i+1} guardada en {filename}')
        i += 1
        start_time = time.time()  # Resetear el tiempo de inicio

    # Esperar que se presione la tecla 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
