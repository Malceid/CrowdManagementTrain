
# Video Processing and Detection Flask App

This Flask application performs real-time video processing and object detection using YOLO models. It supports person detection, violence detection, and dynamic management of video files and cameras. The application also logs detection data and provides options to export logs.


## Features

- Real-time Video Processing: Process and analyze up to 4 videos in real-time
- Object and Behavior Detection: Utilize YOLO models for accurate person and violence detection.
- Logging and Exporting: Log detection data and export logs for further analysis.


## Setup Instructions

Create a Python Virtual Environment To create a virtual environment, run the following command:

`python -m venv <name_of_your_environment>`

Activate the Virtual Environment For Windows: 

`<name_of_your_environment>\Scripts\activate`

For macOS/Linux: 

`source <name_of_your_environment>/bin/activate`

Install the required dependencies using the following commands: 

`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117`

Then, install the remaining dependencies:

`pip install Flask opencv-python-headless ultralytics numpy`

## After Setting Up

For video file processing only run: `python demo-app.py`

For camera usage, run: `python main-app.py`