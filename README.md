# Real-Time Vehicle Speed Estimation

This project uses the YOLOv8 object detection model to detect and track vehicles in a video stream and calculate their real-time speed. It applies a perspective transformation to map pixel coordinates to real-world coordinates for accurate speed estimation.

![Project Demo GIF](https://via.placeholder.com/600x300.png?text=Add+a+GIF+of+your+project+here!)

## 📋 Features
- **Object Detection**: Utilizes a pre-trained YOLOv8x model for robust vehicle detection.
- **Object Tracking**: Employs ByteTrack for consistent tracking of vehicles across frames.
- **Perspective Transformation**: Converts video pixel coordinates to a real-world coordinate system to enable accurate distance measurement.
- **Speed Calculation**: Estimates the speed of each tracked vehicle in km/h based on distance traveled over time.

## 💻 Technologies Used
- **Python 3.x**
- **Ultralytics YOLOv8**
- **Supervision**
- **OpenCV**
- **NumPy**

## 🚀 Setup and Installation

**1. Clone the repository:**
```bash
git clone [https://github.com/Mannan-15/Real-Time-Vehicle-Speed-Estimation.git](https://github.com/Mannan-15/Real-Time-Vehicle-Speed-Estimation.git)
cd Real-Time-Vehicle-Speed-Estimation
```

**2. Install Git LFS:**
This project uses Git LFS to handle the large `yolov8x.pt` model file. You must have Git LFS installed. You can download it from [git-lfs.github.com](https://git-lfs.github.com).

**3. Install dependencies:**
Create a virtual environment (optional) and install the required libraries.
```bash
pip install -r requirements.txt
```

## 使い方 (Usage)
To run the speed estimation on the provided sample video, execute the main script from your terminal:
```bash
python carspeed.py
```
A window will pop up showing the video feed with bounding boxes, tracking IDs, and the calculated speed for each vehicle.
