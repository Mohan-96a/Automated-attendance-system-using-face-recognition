# Face Recognition Attendance System

A sophisticated attendance management system using facial recognition technology, built with Python. The system provides real-time face detection, automatic attendance marking, and email notifications for absent students.

## Features

- **Real-time Face Recognition**: Uses OpenCV and face_recognition library for accurate face detection and recognition
- **Distance Optimization**: Provides real-time feedback for optimal camera distance
- **Automatic Attendance Marking**: Records attendance with timestamps in an Excel sheet
- **Multiple Face Detection Warning**: Alerts when multiple faces are detected in frame
- **Email Notifications**: Automatically sends email alerts to absent students
- **Attendance Status Display**: Shows real-time attendance statistics
- **Excel Integration**: Maintains attendance records in an organized Excel format

## Prerequisites

- Python 3.6+
- OpenCV (`cv2`)
- NumPy
- Pandas
- face_recognition
- smtplib (for email notifications)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
```

2. Install the required dependencies:
```bash
pip install opencv-python numpy pandas face_recognition
```

3. Configure the student data:
   - Add student images to the designated folder
   - Update the `face_data` dictionary with image paths
   - Update the `students` dictionary with student information

4. Configure email settings:
   - Update `sender_email` and `sender_password` in the code
   - Ensure less secure app access is enabled for the sender Gmail account

## Usage

1. Run the main script:
```bash
python attendance_system.py
```

2. Controls:
   - Press 'd' to display current attendance status
   - Press 'q' to quit the application

## System Components

### 1. Face Recognition
- Uses the `face_recognition` library for accurate face detection and recognition
- Compares detected faces with pre-loaded known face encodings
- Provides real-time feedback on face detection status

### 2. Distance Optimization
- Calculates optimal distance between camera and subject
- Provides visual feedback for proper positioning
- Configurable optimal distance and tolerance settings

### 3. Attendance Management
- Automatically marks attendance when a face is recognized at optimal distance
- Records timestamp for each attendance entry
- Maintains daily attendance records in Excel format
- Prevents duplicate attendance entries

### 4. Email Notifications
- Automatically sends email notifications to absent students
- Customizable email templates
- Secure SMTP integration with Gmail

## File Structure

```
attendance_system/
├── attendance_system.py    # Main application file
├── attendance.xlsx        # Attendance records
├── images/               # Student images for face recognition
└── README.md            # Documentation
```

## Configuration

### Student Data Structure
```python
self.students = {
    "STUDENT_NAME": {
        "present": False,
        "time": None,
        "email": "student@email.com",
        "alert_shown": False
    }
}
```

### Face Data Structure
```python
self.face_data = {
    "STUDENT_NAME": "path/to/image.jpg"
}
```

## Best Practices

1. **Image Quality**
   - Use clear, well-lit photos for face recognition
   - Ensure consistent image size and format
   - Update face data regularly

2. **Environment Setup**
   - Maintain good lighting conditions
   - Position camera at eye level
   - Ensure stable internet connection for email notifications

3. **Data Management**
   - Regularly backup attendance records
   - Maintain student information confidentiality
   - Update student data promptly

## Troubleshooting

1. **Camera Issues**
   - Ensure camera is properly connected
   - Check camera permissions
   - Verify camera index in `cv2.VideoCapture()`

2. **Face Recognition Problems**
   - Check image quality and lighting
   - Verify face data paths
   - Adjust face recognition tolerance if needed

3. **Email Notification Issues**
   - Verify email credentials
   - Check internet connection
   - Ensure proper SMTP configuration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- OpenCV team for computer vision capabilities
- face_recognition library developers
- Python community for various dependencies
