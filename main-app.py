from flask import Flask, render_template, Response, redirect, url_for, send_file, request
import csv
import datetime
import threading
import cv2
from ultralytics import YOLO
import logging

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)

try:
    # Load YOLO models
    person_model = YOLO('models/yolov8s.pt')
    violence_model = YOLO('models/violence2.pt')
except Exception as e:
    logging.error(f"Error loading YOLO models: {e}")
    raise

# List of camera device indices
camera_indices = [0, 1, 2, 3]  # Adjust this list based on the number of cameras you have

# Initialize video capture objects dictionary
video_captures = {i: cv2.VideoCapture(camera_indices[i]) for i in range(len(camera_indices))}

# Initialize log data and threading lock
log_data = []
log_lock = threading.Lock()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/main_page')
def main_page():
    return render_template('main.html')

def process_frame(frame, camera_number):
    frame = cv2.resize(frame, (1020, 500))
    try:
        # Run YOLOv8s (person detection) inference on the frame
        results_person = person_model(frame, stream=True, device=0)
        person_count = 0

        # Process person detections
        for result in results_person:
            for box in result.boxes:
                if person_model.names[int(box.cls)] == "person":
                    person_count += 1

        # Run YOLOv8 violence model inference on the frame
        results_violence = violence_model(frame, stream=True, device=0)
        violence_detected = False

        # Process violence detections
        for result in results_violence:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls)
                if violence_model.names[class_id] == "violence":
                    color = (0, 0, 255)  # Red for violence
                    violence_detected = True
                else:
                    color = (255, 0, 0)  # Blue for non-violence

                # Draw bounding box
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = 'Violence' if violence_model.names[class_id] == "violence" else 'No Violence'
                frame = cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Log the data
        with log_lock:
            log_data.append({
                'timestamp': datetime.datetime.now().isoformat(),
                'person_count': person_count,
                'violence_detected': violence_detected,
                'camera_number': camera_number
            })

        person_text = f'Person Count: {person_count}'

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        border_thickness = font_thickness + 2

        # Define text position
        position = (10, 68)

        # Draw the text with a white border
        frame = cv2.putText(frame, person_text, position, font, font_scale, (255, 255, 255), border_thickness, cv2.LINE_AA)

        # Draw the text in red on top
        frame = cv2.putText(frame, person_text, position, font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)

        return frame
    except Exception as e:
        logging.error(f"Error processing frame: {e}")
        return frame

def generate_frames(camera_index):
    while True:
        if camera_index not in video_captures:
            break

        cap = video_captures[camera_index]
        success, frame = cap.read()

        if success:
            frame = process_frame(frame, camera_index)
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # Yield frame to display
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            # Handle case where frame is not read successfully
            continue

@app.route('/video_feed')
def video_feed():
    try:
        camera_number = int(request.args.get('camera_number', 0))
        if camera_number not in video_captures:
            return "Video stream not found", 404
        return Response(generate_frames(camera_number), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logging.error(f"Error in video_feed: {e}")
        return str(e), 500

@app.route('/add_camera/<int:camera_index>')
def add_camera(camera_index):
    global video_captures
    try:
        if 0 <= camera_index < len(camera_indices):
            if camera_index not in video_captures:
                video_captures[camera_index] = cv2.VideoCapture(camera_indices[camera_index])
            return f"Camera {camera_index} added successfully", 200
        else:
            return "Camera index out of range", 404
    except Exception as e:
        logging.error(f"Error in add_camera: {e}")
        return str(e), 500

@app.route('/view_all_videos')
def view_all_videos():
    return redirect(url_for('main_page'))

@app.route('/export_log')
def export_log():
    global log_data, log_lock
    csv_file = "activity_log.csv"
    csv_columns = ['timestamp', 'person_count', 'violence_detected', 'camera_number']

    try:
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            with log_lock:
                for data in log_data:
                    writer.writerow(data)
        return send_file(csv_file, as_attachment=True)
    except Exception as e:
        logging.error(f"Error in export_log: {e}")
        return str(e), 500

if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        logging.error(f"Error running app: {e}")
        raise
