"""get_mesurments.py

    Author: Alberto Castro
    Organisation: Universidad de Monterrey
    Contact: josealberto.castro@udem.edu

    EXAMPLE OF USAGE:
    python3 get_mesurments.py --cam_index 0 --Z 40 --cal_file calibration-parameters/calibration_data_laptop.json
"""

import cv2
import numpy as np
import argparse
import json
import os
from typing import List, Tuple
import sys

# Initialize global variables
points = []  # This will store the real-world coordinates
pixel_points = []  # This will store the pixel coordinates
frame = None
left_click_block = False

def parse_data_from_cli_get_measurements():
    """
    Parse data from the command line interface (CLI) for the get-measurements.py script.

    Returns:
        args: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Get measurements from a camera image')
    parser.add_argument('--cam_index', type=int, required=True, help='Index of the camera to use')
    parser.add_argument('--Z', type=float, required=True, help='Distance from the camera to the object in cm')
    parser.add_argument('--cal_file', type=str, required=True, help='Path to the camera calibration file')
    return parser.parse_args()

def load_calibration_parameters_from_json_file(cal_file: str):
    """
    Load camera calibration parameters from a JSON file.

    Args:
        cal_file: Path to the camera calibration file.

    Returns:
        camera_matrix: Camera matrix.
        distortion_coefficients: Distortion coefficients.

    This function may raise a warning if the JSON file 
    does not exist. In such a case, the program finishes.
    """
    if not os.path.isfile(cal_file):
        print(f"The file {cal_file} does not exist!")
        sys.exit(-1)
    with open(cal_file) as f:
        json_data = json.load(f)
    camera_matrix = np.array(json_data['camera_matrix'])
    distortion_coefficients = np.array(json_data['distortion_coefficients'])
    return camera_matrix, distortion_coefficients

def compute_real_coordinates(points: List[Tuple[int, int]], Z: float, fx: float, fy: float, cx: float, cy: float) -> List[Tuple[float, float]]:
    """
    This function computes the real-world coordinates of a set of points in the image.

    Args:
        points: List of pixel coordinates.
        Z: Distance from the camera to the object in cm.
        fx: Focal length in x.
        fy: Focal length in y.
        cx: Principal point in x.
        cy: Principal point in y.

    Returns:
        real_coordinates: List of real-world coordinates.
    """
    real_coordinates = []
    for (ui, vi) in points:
        X = (ui - cx) * Z / fx
        Y = (vi - cy) * Z / fy
        real_coordinates.append((X, Y))
    return real_coordinates

def compute_line_segments(real_coordinates: List[Tuple[float, float]]) -> List[float]:
    """
    This function computes the length of each line segment in a set of real-world coordinates.

    Args:
        real_coordinates: List of real-world coordinates.

    Returns:
        line_segments: List of line segments.
    """
    line_segments = [np.linalg.norm(np.array(real_coordinates[i]) - np.array(real_coordinates[i + 1]))
                     for i in range(len(real_coordinates) - 1)]
    return line_segments

def compute_perimeter(line_segments: List[float]) -> float:
    """
    This function computes the perimeter of a closed figure given the lengths of its line segments.

    Args:
        line_segments: List of line segments.

    Returns:
        perimeter: Perimeter of the closed figure.
    """

    return sum(line_segments)

def mouse_callback(event, x, y, flags, param):
    """
    This function handles mouse events for the 'Frame' window.

    Args:
        event: Mouse event.
        x: x-coordinate of the mouse pointer.
        y: y-coordinate of the mouse pointer.
        flags: Flags.
        param: Parameters.

    Returns:
        None
    """
    global left_click_block, points, pixel_points, camera_matrix, Z

    if camera_matrix is not None:
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

        if event == cv2.EVENT_LBUTTONDOWN and not left_click_block:
            # Calculate real-world coordinate and append to points
            real_x = (x - cx) * Z / fx
            real_y = (y - cy) * Z / fy
            points.append((real_x, real_y))
            pixel_points.append((x, y))  # Also save pixel point for drawing

            if len(points) > 1:
                # Only calculate distance for the last two points added
                last_segment_length = np.linalg.norm(np.array(points[-1]) - np.array(points[-2]))
                print(f'Distancia del segmento actual: {last_segment_length}')

        elif event == cv2.EVENT_MBUTTONDOWN and len(points) > 1:
            # Append the first point to close the figure and calculate distances
            closed_points = points + [points[0]]
            closed_pixel_points = pixel_points + [pixel_points[0]]
            real_coords = compute_real_coordinates(closed_pixel_points, Z, fx, fy, cx, cy)
            line_segments = compute_line_segments(real_coords)
            perimeter = compute_perimeter(line_segments)
            # Sort distances from longest to shortest and print
            line_segments_sorted = sorted(line_segments, reverse=True)
            print(f'Perímetro de la figura cerrada: {perimeter}')
            print(f'Distancias de segmentos de mayor a menor: {line_segments_sorted}')
            left_click_block = True

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Remove the last point both from real and pixel coordinates
            if len(points) > 0:
                points.pop()
                pixel_points.pop()
            left_click_block = False

        elif flags == cv2.EVENT_FLAG_CTRLKEY and event == cv2.EVENT_LBUTTONDOWN:
            # Reset everything
            points.clear()
            pixel_points.clear()
            left_click_block = False


def initialize_camera(cam_index: int) -> cv2.VideoCapture:
    """
    This function initializes the camera with the given index.

    Args:
        cam_index: Index of the camera to initialize.

    Returns:
        cap: VideoCapture object.
    """
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise IOError(f'Cannot open camera with index {cam_index}')
    return cap

def run_pipeline(cam_index: int, cal_file: str, Z: float):
    """
    This function runs the pipeline to get measurements from a camera image.

    Args:
        cam_index: Index of the camera to use.
        cal_file: Path to the camera calibration file.
        Z: Distance from the camera to the object in cm.

    Returns:
        None
    """
    global points, pixel_points, frame, camera_matrix, distortion_coefficients
    camera_matrix, distortion_coefficients = load_calibration_parameters_from_json_file(cal_file)

    cap = initialize_camera(cam_index)
    cv2.namedWindow('Frame')
    cv2.setMouseCallback('Frame', mouse_callback)

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            print('Error capturing image from camera.')
            break

        frame = cv2.undistort(raw_frame, camera_matrix, distortion_coefficients)
        for i, point in enumerate(pixel_points):
            cv2.circle(frame, point, 5, (0, 255, 0), -1)
            if i > 0:
                cv2.line(frame, pixel_points[i - 1], pixel_points[i], (255, 0, 0), 2)
        if left_click_block and len(pixel_points) > 1:
            cv2.line(frame, pixel_points[-1], pixel_points[0], (255, 0, 0), 2)  # Close the figure

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    """
    This function is the main entry point of the script. It parses command-line arguments using
    'parse_data_from_cli_get_measurements' function, then runs the camera calibration pipeline
    using 'run_pipeline' function.
    """
    global Z
    args = parse_data_from_cli_get_measurements()
    Z = args.Z
    run_pipeline(args.cam_index, args.cal_file, args.Z)

if __name__ == "__main__":
    main()