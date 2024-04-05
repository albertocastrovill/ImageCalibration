"""get_mesurments.py

    Author: Alberto Castro
    Organisation: Universidad de Monterrey
    Contact: josealberto.castro@udem.edu

    EXAMPLE OF USAGE:
    python3 get_mesurments.py --cam_index 0 --Z 40 --cal_file calibration-parameters/calibration_data_laptop.json
"""

# Import standard libraries
import cv2
import numpy as np
import argparse
import json
import os
from typing import List, Tuple
import sys

points = []
frame = None

def parse_data_from_cli_get_measurements()->argparse.ArgumentParser:
    """
    Parse data from the command line interface (CLI) for the get-measurements.py script.

    Returns:
        args: Parsed command-line arguments.
    """
    
    # Create a parser object
    parser = argparse.ArgumentParser(description='Get measurements from a camera image')
    
    # Define the command line arguments
    parser.add_argument('--cam_index', type=int, required=True, help='Index of the camera to use')
    parser.add_argument('--Z', type=int, required=True, help='Distance from the camera to the object in cm')
    parser.add_argument('--cal_file', type=str, required=True, help='Path to the camera calibration file')
    
    # Parse the arguments
    args = parser.parse_args()
    
    return args

def load_calibration_parameters_from_json_file(
        cal_file: str
        )->None:
    """
    Load camera calibration parameters from a JSON file.

    Args:
        args: Parsed command-line arguments.

    Returns:
        camera_matrix: Camera matrix.
        distortion_coefficients: Distortion coefficients.

    This function may raise a warning if the JSON file 
    does not exist. In such a case, the program finishes.
    """

    # Check if JSON file exists
    json_filename = cal_file
    check_file = os.path.isfile(json_filename)

    # If JSON file exists, load the calibration parameters
    if check_file:
        f = open(json_filename)
        json_data = json.load(f)
        f.close()
        
        camera_matrix = np.array(json_data['camera_matrix'])
        distortion_coefficients = np.array(json_data['distortion_coefficients'])
        return camera_matrix, distortion_coefficients
    
    # Otherwise, the program finishes
    else:
        print(f"The file {json_filename} does not exist!")
        sys.exit(-1)

"""
#Función de selección de puntos usando el mouse
def mouse_callback(event, x, y, flags, param):
    """
"""
    Callback function for mouse events.

    Args:
        event: Mouse event.
        x: X-coordinate of the mouse pointer.
        y: Y-coordinate of the mouse pointer.
        flags: Flags for the mouse event.
        param: Additional parameters for the mouse event.

    Returns:
        None: The function does not return any value.
    """
"""
    # Get the global variables
    global left_click_block
    global wh
    global points  
    global coords  
    global lines   

    i = len(points)
    
    if event == cv2.EVENT_LBUTTONDOWN and not left_click_block:
        print('Punto', i, 'seleccionado')
        points.append([x, y])

    elif event == cv2.EVENT_MBUTTONDOWN:
        if not left_click_block:
            if len(points) > 1:
                print('Calculando distancia...')
                left_click_block = True
                wh = True
                
            else:
                print('No se han seleccionado suficientes puntos')

    elif event == cv2.EVENT_RBUTTONDOWN and left_click_block:
        # finalizar la selección con botón derecho
        print('Finalizando selección...')
        left_click_block = False
        wh = False
        # añadir acciones a realizar tras finalizar la selección

    elif flags == cv2.EVENT_FLAG_CTRLKEY:
        # limpiar todo con una tecla especial, aquí CTRL
        print('Reiniciando puntos...')
        points.clear()
        coords.clear()
        lines.clear()
        left_click_block = False
        wh = False
"""

def calculate_distances_and_perimeter(points: List[Tuple[int, int]]) -> Tuple[List[float], float]:
    """
    Calculate the distances between consecutive points and the perimeter of the closed figure.

    Args:
        points: List of points where each point is represented as (x, y).

    Returns:
        distances: List of distances between consecutive points.
        perimeter: Total perimeter of the closed figure.
    """
    distances = []
    n = len(points)
    if n > 1:
        for i in range(n - 1):
            dx = points[i+1][0] - points[i][0]
            dy = points[i+1][1] - points[i][1]
            distances.append(np.sqrt(dx**2 + dy**2))
        # Close the figure by connecting the last point to the first
        dx = points[0][0] - points[-1][0]
        dy = points[0][1] - points[-1][1]
        distances.append(np.sqrt(dx**2 + dy**2))
        perimeter = sum(distances)
    else:
        perimeter = 0
    return distances, perimeter

left_click_block = False 
wh = False

# Update the mouse_callback function
def mouse_callback(event, x, y, flags, param):
    global left_click_block, points

    if event == cv2.EVENT_LBUTTONDOWN and not left_click_block:
        print(f'Punto {len(points)} seleccionado: ({x}, {y})')
        points.append((x, y))

    elif event == cv2.EVENT_MBUTTONDOWN and len(points) > 1:
        print('Calculando distancia y cerrando figura...')
        distances, perimeter = calculate_distances_and_perimeter(points)
        left_click_block = True
        print(f'Distancias: {distances}')
        print(f'Perímetro: {perimeter}')
        # Re-drawing logic will be handled in the main loop

    # Eliminar el último punto
    elif event == cv2.EVENT_RBUTTONDOWN:
        #Eliminar el último punto
        if len(points) > 0:
            print('Eliminando último punto...')
            points.pop()
            left_click_block = False
        else:
            print('No hay puntos para eliminar.')

    elif flags == cv2.EVENT_FLAG_CTRLKEY:
        print('Reiniciando puntos...')
        points.clear()
        left_click_block = False

    
# Initialize camera
def initialize_camera(cam_index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise IOError(f'No se puede abrir la cámara con índice {cam_index}')
    return cap

# Run the pipeline
def run_pipeline(cam_index: int, cal_file: str):
    global points
    camera_matrix, distortion_coefficients = load_calibration_parameters_from_json_file(cal_file)
    cap = initialize_camera(cam_index)

    cv2.namedWindow('Frame')
    cv2.setMouseCallback('Frame', mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Error al capturar imagen de la cámara.')
            break

        # Dibuja los puntos y líneas en el frame
        for i, point in enumerate(points):
            cv2.circle(frame, point, 5, (0, 255, 0), -1)
            if i > 0:
                cv2.line(frame, points[i - 1], points[i], (255, 0, 0), 2)
        if left_click_block and len(points) > 1:
            cv2.line(frame, points[-1], points[0], (255, 0, 0), 2)  # Cierra la figura

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function
def main():
    args = parse_data_from_cli_get_measurements()
    run_pipeline(args.cam_index, args.cal_file)

if __name__ == "__main__":
    main()


