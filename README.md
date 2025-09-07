# Image Classification Web App with YOLOv8

A web application that performs object detection on uploaded images using the YOLOv8 model from Ultralytics.

## Features

- Drag and drop image upload
- Real-time object detection
- Bounding box visualization
- Mobile-responsive design
- Clean and intuitive user interface

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/image-classification-app.git
   cd image-classification-app
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Flask application:
   ```bash
   python3 app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Upload an image by dragging and dropping it onto the upload area or by clicking to select a file.

4. View the detection results with bounding boxes and class probabilities.

## Project Structure

```
.
├── app.py                # Main Flask application
├── requirements.txt      # Python dependencies
├── static/               # Static files (CSS, JS, uploaded images)
│   └── uploads/          # Directory for uploaded images
└── templates/            # HTML templates
    └── index.html        # Main web interface
```

## Customization

- To use a different YOLOv8 model, modify the model name in `app.py` (e.g., 'yolov8s.pt', 'yolov8m.pt', etc.)
- Adjust the confidence threshold in the JavaScript code to filter detections
- Customize the UI by modifying the CSS in `templates/index.html`

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Flask](https://flask.palletsprojects.com/)
- [Tailwind CSS](https://tailwindcss.com/)
# classify-object
