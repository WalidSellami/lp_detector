######## Test Code ########



import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR


model_path = '/content/drive/MyDrive/license_detector_model.pt'
model = YOLO(model_path)

# Load OCR
ocr = PaddleOCR(lang='en')


# Initialize the camera (0 for default camera, change index for other cameras)
cap = cv2.VideoCapture(0)

# Define codec and create VideoWriter to save the output video (optional, for recording real-time output)
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (640, 480))

# Ensure camera is opened
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Process frames from the camera in real-time
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Run the license plate detector on the frame
    results = model(frame)
    results_boxes = results[0].boxes.data.tolist()

    if results_boxes != []:
    # Process detection results
       for result in results:
          if result is not None:
              for box in result.boxes:
                  x1, y1, x2, y2 = map(int, box.xyxy[0])
                  conf = box.conf[0]

                  if conf >= 0.6:  # Confidence threshold
                      # Crop the license plate region from the frame
                      license_plate_image = frame[y1:y2, x1:x2]

                      # Convert to grayscale
                      gray = cv2.cvtColor(license_plate_image, cv2.COLOR_BGR2GRAY)

                      # Apply bilateral filter for noise reduction
                      filtered = cv2.bilateralFilter(gray, 3, 25, 25)

                      # Apply adaptive histogram equalization (CLAHE)
                      clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                      enhanced_image = clahe.apply(filtered)

                      # OCR detection on cropped plate region
                      result_ocr = ocr.ocr(enhanced_image, det=True, rec=True)

                      license_plate_number = ''
                      if result_ocr != [None]:
                          for line in result_ocr:
                              for element in line:
                                  license_plate_number = element[1][0]
                                  license_plate_number = license_plate_number.replace(' ', '')

                                  # Draw bounding box and text on the frame
                                  cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                                  # Draw a filled rectangle for the text background
                                  text_size = cv2.getTextSize(license_plate_number, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                                  text_bg_x1 = x1
                                  text_bg_y1 = y1 - text_size[1] - 10
                                  text_bg_x2 = x1 + text_size[0] + 5
                                  text_bg_y2 = y1

                                  cv2.rectangle(frame, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), (0, 0, 0), -1)
                                  cv2.putText(frame, license_plate_number.strip(), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                                  print(f"License plate found: {license_plate_number}")

          else:
              print('No license plate detected in the frame ...')
              
    else:
        print('No license plate detected ... :(')
        
        
    # Display the frame
    cv2.imshow("License Plate Detection", frame)

    # Write the processed frame to the output video
    # out.write(frame)

    # Press 'q' to exit the real-time processing
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video objects and close windows
cap.release()
# out.release()
cv2.destroyAllWindows()