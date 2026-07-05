# 🧍 Object Detection using OpenCV

A Computer Vision project built using **Python** and **OpenCV** that detects people in images using the **Histogram of Oriented Gradients (HOG) Descriptor**. The program identifies human figures, draws bounding boxes around detected people, counts the total number of detections, and saves the processed image automatically

## 📌 Features

- Detects people in images using OpenCV's HOG Descriptor
- Uses OpenCV's pre-trained SVM People Detector
- Draws bounding boxes around detected people
- Displays the total number of people detected
- Saves the output image automatically
- Beginner-friendly and easy to understand
- Lightweight implementation with no custom model training

## 🛠️ Technologies Used

- Python 3.x
- OpenCV
- NumPy
- Imutils



## ▶️ How to Run

Place your image inside the **images** folder.

Run the program:

```bash
python detection.py
```

---

## 📸 Output

The program will:

- Detect people in the image
- Draw green bounding boxes
- Display the number of detected people
- Save the processed image inside the **outputs** folder

Example output:

```
=============================================
      OBJECT DETECTION COMPLETED
=============================================
People Detected : 3
=============================================
Output image saved at:
outputs/result.jpg
```

## 🧠 Working Principle

1. Read the input image.
2. Resize the image for faster processing.
3. Initialize the HOG Descriptor.
4. Load OpenCV's pre-trained People Detector.
5. Detect people in the image.
6. Apply Non-Maximum Suppression (NMS) to remove duplicate detections.
7. Draw bounding boxes and labels.
8. Display the people count.
9. Save and display the final output image.

---



## 🚀 Future Improvements

- Real-time webcam detection
- Video object detection
- Detection confidence display
- Support for multiple object classes
- Object tracking
- Deep learning-based detection using YOLO

---

## 📋 Requirements

```
opencv-python
numpy
imutils
```

---

## 👨‍💻 Author

**Tegnoor Singh**

B.Tech Artificial Intelligence & Machine Learning

CGC University, Mohali

---

## 📄 License

This project is licensed under the MIT License.

---

⭐ If you found this project helpful, consider giving it a star!
