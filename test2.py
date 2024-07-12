import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from tensorflow.keras.models import load_model

# Error handling (optional)
try:
    model = load_model("Model/keras_model.h5")
except OSError:
    print("Error loading model. Please check the path and file existence.")
    exit()

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")  # Corrected path
offset = 20
imgSize = 300
labels = ["0", "1", "2", "3"]  # Update with your actual class names

# FPS calculation (optional)
start_time = time.time()
frames_processed = 0

while True:
    success, img = cap.read()
    imgOutput = img.copy()

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape
        aspectRatio = h / w

        # Preprocessing based on aspect ratio
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        prediction, index = classifier.getPrediction(imgWhite, draw=False)

        # Draw rectangle, label, and predicted probability (optional)
        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, f"{labels[index]} ({prediction:.2f})", (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

        # Display additional images for debugging (optional)
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    # Display main image and calculated FPS (optional)
    cv2.imshow("Image", imgOutput)

    # FPS calculation
    frames_processed += 1
    current_time = time.time()
    fps = frames_processed / (current_time - start_time)
    cv2.putText(imgOutput, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
