"""get_mesurments_1.py

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

def compute_real_coordinates(points: List[Tuple[int, int]], Z, fx, fy, cx, cy) -> List[Tuple[float, float]]:
    """
    Compute the real-world coordinates of the points using the provided formulas.

    Args:
        points (List[Tuple[int, int]]): List of points where each point is represented as (x, y).
        Z (float): Distance from the camera to the object plane.
        fx (float): Focal length of the camera along the x axis.
        fy (float): Focal length of the camera along the y axis.
        cx (float): Horizontal pixel coordinate of the image center.
        cy (float): Vertical pixel coordinate of the image center.

    Returns:
        List[Tuple[float, float]]: List of real-world coordinates.
    """
    real_coordinates = []
    for (ui, vi) in points:
        X = (ui - cx) * Z / fx
        Y = (vi - cy) * Z / fy
        real_coordinates.append((X, Y))
    return real_coordinates

def compute_line_segments(real_coordinates: List[Tuple[float, float]]) -> List[float]:
    """
    Calculate the length of each line segment between consecutive real-world points.

    Args:
        real_coordinates (List[Tuple[float, float]]): List of real-world coordinates where each point is represented as (X, Y).

    Returns:
        List[float]: Lengths of the line segments.
    """
    line_segments = [np.linalg.norm(np.array(real_coordinates[i]) - np.array(real_coordinates[i + 1]))
                     for i in range(len(real_coordinates) - 1)]
    
    print(f'Distancias: {line_segments}')
    return line_segments


def compute_perimeter(line_segments: List[float]) -> float:
    """
    Compute the perimeter of the shape formed by the line segments.

    Args:
        line_segments (List[float]): Lengths of the line segments.

    Returns:
        float: Total perimeter of the shape.
    """
    perimetro = sum(line_segments)
    print(f'Perímetro: {perimetro}')
    return perimetro


left_click_block = False 
wh = False

# Update the mouse_callback function
def mouse_callback(event, x, y, flags, param):
    global left_click_block, points

    if event == cv2.EVENT_LBUTTONDOWN and not left_click_block:
        print(f'Punto {len(points)} seleccionado: ({x}, {y})')
        points.append((x, y))

    elif event == cv2.EVENT_MBUTTONDOWN and len(points) > 1:
        # Cerrar la figura conectando el último punto con el primero
        closed_points = points + [points[0]]
        print('Calculando distancia y cerrando figura...')
        line_segments = compute_line_segments(closed_points)
        perimeter = compute_perimeter(line_segments)
        left_click_block = True
        print(f'Distancias: {line_segments}')
        print(f'Perímetro: {perimeter}')

        # Ordenar las distancias de los segmentos de línea de mayor a menor
        line_segments_desc = sorted(line_segments, reverse=True)
        print(f'Distancias de segmentos de mayor a menor: {line_segments_desc}')

    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(points) > 0:
            print('Eliminando último punto...')
            points.pop()
            left_click_block = False  # Permitir la selección de puntos de nuevo
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
def run_pipeline(cam_index: int, cal_file: str, Z: float):
    global points, frame
    camera_matrix, distortion_coefficients = load_calibration_parameters_from_json_file(cal_file)

    # Extraer fx, fy, cx, cy de camera_matrix
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    cap = initialize_camera(cam_index)

    cv2.namedWindow('Frame')
    cv2.setMouseCallback('Frame', mouse_callback)

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            print('Error al capturar imagen de la cámara.')
            break

        # Aplicar la corrección de distorsión a la imagen capturada
        frame = cv2.undistort(raw_frame, camera_matrix, distortion_coefficients)

        # Dibuja los puntos y líneas en el frame corregido
        for i, (x, y) in enumerate(points):
            # Convierte las coordenadas de píxeles para el dibujo
            pixel_coords = (int(x), int(y))
            cv2.circle(frame, pixel_coords, 5, (0, 255, 0), -1)
            if i > 0:
                cv2.line(frame, points[i - 1], pixel_coords, (255, 0, 0), 2)

        if left_click_block and len(points) > 1:
            # Asegúrate de cerrar la figura
            cv2.line(frame, points[-1], points[0], (255, 0, 0), 2)  # Cierra la figura
            closed_points = points + [points[0]]
            real_coords = compute_real_coordinates(closed_points, Z, fx, fy, cx, cy)
            line_segments = compute_line_segments(real_coords)
            perimeter = compute_perimeter(line_segments)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Main function
def main():
    args = parse_data_from_cli_get_measurements()
    run_pipeline(args.cam_index, args.cal_file, args.Z)

if __name__ == "__main__":
    main()


