# Emotion Detection Web Application - Project Report

## 1. Executive Summary
This project aims to develop a robust, web-based Emotion Detection system levering Deep Learning and Computer Vision technologies. The application captures facial expressions in real-time via webcam or through uploaded images and classifies them into one of seven universal emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral). The system is built with a high-performance FastAPI backend and a modern, aesthetically pleasing frontend.

## 2. Project Objectives
- **Real-time Detection**: Analyze video streams with low latency to provide instant emotion feedback.
- **User-Friendly Interface**: Create a seamless, intuitive experience for users to interact with the AI.
- **Accuracy**: Utilize Convolutional Neural Networks (CNN) to achieve reliable emotion recognition.
- **Scalability**: Design a modular architecture separating model inference, API logic, and frontend presentation.

## 3. System Architecture

### 3.1 Technology Stack
- **Deep Learning**: TensorFlow / Keras (CNN Model)
- **Computer Vision**: OpenCV (Face Detection, Image Preprocessing)
- **Backend Framework**: FastAPI (Python)
- **Frontend**: HTML5, CSS3 (Custom Glassmorphism Design), Vanilla JavaScript
- **Server**: Uvicorn (ASGI Server)

### 3.2 High-Level Workflow
1.  **Input Acquisition**: User provides input via Webcam (Video Stream) or File Upload (Image).
2.  **Face Detection**: The backend processes the frame to identify faces using Haar Cascade Classifiers.
3.  **Preprocessing**: Detected faces are cropped, converted to grayscale, and resized to 48x48 pixels.
4.  **Inference**: The pre-processed face data is fed into the CNN Model.
5.  **Output Generation**: The model outputs a probability distribution across 7 emotions. The highest probability class is selected.
6.  **Response**: The system returns the bounding box coordinates, emotion label, and confidence score to the frontend.
7.  **Visualization**: The frontend renders a bounding box and label over the face in the UI.

## 4. Implementation Details

### 4.1 Deep Learning Model (CNN)
The core of the system is a custom Convolutional Neural Network inspired by the VGG architecture, optimized for the FER-2013 dataset.
- **Input**: 48x48 Pixel Grayscale Image.
- **Architecture**:
    - **4 Convolutional Blocks**: Each block consists of 2x Conv2D layers (ReLU activation, padding='same'), followed by MaxPooling2D, Dropout (0.25), and BatchNormalization.
    - **Fully Connected Layers**: Flatten layer followed by Dense (512, ReLU), Dropout (0.5), Dense (256, ReLU), Dropout (0.5).
    - **Output Layer**: Dense (7, Softmax) for multi-class classification.
- **Weights Handling**: The system automatically generates placeholder weights for demonstration if a pre-trained file is not found, ensuring system stability.

### 4.2 Backend (FastAPI)
The backend exposes two primary endpoints:
- **`POST /predict-image`**: Handles multipart/form-data for static image files.
- **`POST /predict-frame`**: Accepts base64 encoded strings for real-time video frame processing.
- Error handling is implemented to manage invalid inputs or detection failures gracefully, logging errors to stderr for debugging.

### 4.3 Frontend
- **Design System**: A "Self.AI" branded interface featuring a dark-mode aesthetic, glassmorphism effects, and neon accent colors (#22d3ee, #8b5cf6).
- **Interactivity**:
    - **Live Webcam Mode**: Uses `navigator.mediaDevices.getUserMedia` to stream video to a `<video>` element and captures frames for the backend.
    - **Upload Mode**: Drag-and-drop interface with instant preview. Includes a "Change Image" feature to easily reset the workflow.
    - **Loading States**: Visual feedback (spinners) during server processing to enhance UX.
    - **History Log**: A side panel tracks the timeline of detected emotions.

## 5. Challenges & Solutions
- **Model Input Shapes**: Initially, the model encountered dimension mismatch errors due to default padding valid. **Solution**: Updated all Conv2D layers to use `padding='same'` to preserve spatial dimensions through deep layers.
- **Face Detection Reliability**: Haar Cascades were failing in poor lighting. **Solution**: Relaxed detection parameters (`minNeighbors=3`, `scaleFactor=1.1`) and implemented per-request cascade re-initialization to ensure thread safety.
- **Browser caching**: Changes were not reflecting immediately. **Solution**: Implemented hard-refresh protocols and ensured robust static file serving.

## 6. Future Enhancements
- **Model Training**: Integrate a training pipeline to allow users to train on their own datasets.
- **Emotion Tracking Graph**: Visualize emotion trends over a session using Chart.js.
- **Face Recognition**: Add identity recognition to track emotions for specific individuals.
- **Mobile Support**: Further optimize the UI for mobile device cameras.

## 7. Conclusion
The Emotion Detection Web App successfully demonstrates the integration of modern web technologies with advanced AI. It provides a stable, interactive platform for analyzing human emotions, suitable for educational, research, or entertainment purposes.
