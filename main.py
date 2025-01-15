import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import threading
import time

# Global variables for email cooldown
last_email_time = 0  # Timestamp of the last email sent
cooldown_period = 60  # Cooldown period in seconds (e.g., 60 seconds = 1 minute)

def send_email(receiver_email, frame):
    """
    Sends an email with the detected fire/smoke frame as an attachment.
    
    Args:
        receiver_email (str): Email address of the recipient.
        frame (numpy.ndarray): The frame/image where fire/smoke was detected.
    """
    try:
        # Set up the SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)  # Use Gmail's SMTP server
        server.starttls()  # Enable secure connection
        server.login("amarkingstone20@gmail.com", "guqa wans fqjl mnpm")  # Update with your credentials , "your_email@gmail.com" , "Your Password"
        # use this link to create temporary password --> https://myaccount.google.com/apppasswords
        # Create the email message
        msg = MIMEMultipart()
        msg['Subject'] = 'Fire/Smoke Detected'
        msg['From'] = "amarkingstone20@gmail.com"  # Update with your email
        msg['To'] = receiver_email

        # Encode the frame as a JPEG image
        _, buffer = cv2.imencode('.jpg', frame)
        img_data = buffer.tobytes()  # Convert the buffer to bytes

        # Attach the image to the email
        img = MIMEImage(img_data, name="Fire_Smoke.jpg")
        img.add_header('Content-Disposition', 'attachment', filename="Fire_Smoke.jpg")
        msg.attach(img)

        # Send the email
        server.send_message(msg)
        print("Email sent successfully.")

    except Exception as e:
        print(f"Failed to send email: {e}.")
    finally:
        try:
            server.quit()  # Terminate the SMTP session
        except:
            pass  # If server was not initialized, just pass


# Load the YOLOv11 model for fire and smoke detection with segmentation
fire_smoke_model = YOLO("best.pt")  
class_names = fire_smoke_model.model.names  # Get the class names from the model

# Open the video file or webcam for processing
video_capture = cv2.VideoCapture('sample.mp4')
total_frames_processed = 0  # Counter to track the number of frames processed

# List to keep track of all email-sending threads
email_threads = []

while True:
    # Read a frame from the video
    ret, current_frame = video_capture.read()
    if not ret:
        break  # Exit the loop if no more frames are available

    # Resize the frame for better display
    current_frame = cv2.resize(current_frame, (1020, 500))

    # Run YOLOv11 segmentation on the frame
    segmentation_results = fire_smoke_model.track(current_frame, persist=True)

    # Check if any objects (e.g., fire or smoke) are detected
    if segmentation_results[0].boxes is not None:
        # Extract bounding boxes, class IDs, and tracking IDs
        bounding_boxes = segmentation_results[0].boxes.xyxy.int().cpu().tolist()
        detected_class_ids = segmentation_results[0].boxes.cls.int().cpu().tolist()

        # Check if tracking IDs exist
        if segmentation_results[0].boxes.id is not None:
            tracking_ids = segmentation_results[0].boxes.id.int().cpu().tolist()
        else:
            tracking_ids = [-1] * len(bounding_boxes)  # Use -1 for objects without IDs

        # Extract segmentation masks (if available)
        segmentation_masks = segmentation_results[0].masks
        if segmentation_masks is not None:
            segmentation_masks = segmentation_masks.xy
            overlay_frame = current_frame.copy()  # Create a copy of the frame for overlay

            # Process each detected object
            for box, track_id, class_id, mask in zip(bounding_boxes, tracking_ids, detected_class_ids, segmentation_masks):
                class_name = class_names[class_id]  # Get the class name (e.g., "fire" or "smoke")
                x1, y1, x2, y2 = box  # Extract bounding box coordinates

                # Check if the mask is not empty
                if mask.size > 0:
                    mask = np.array(mask, dtype=np.int32).reshape((-1, 1, 2))  # Reshape mask for drawing

                    # Draw the bounding box and segmentation mask on the frame
                    cv2.rectangle(current_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box
                    cv2.fillPoly(overlay_frame, [mask], color=(0, 0, 255))  # Red mask for fire/smoke

                    # Display the tracking ID and class label
                    cvzone.putTextRect(current_frame, f'{track_id}', (x2, y2), 1, 1)
                    cvzone.putTextRect(current_frame, f'{class_name}', (x1, y1), 1, 1)

                    # If fire is detected, send an email alert (with cooldown)
                    if ("fire" in class_name.lower()) and (time.time() - last_email_time) > cooldown_period:
                        receiver_email = "amarkingstone20@gmail.com"  # Update with the actual email address
                        email_thread = threading.Thread(target=send_email, args=(receiver_email, current_frame.copy()))
                        email_threads.append(email_thread)  # Track the thread
                        email_thread.start()  # Start the email-sending thread
                        last_email_time = time.time()  # Update the last email time

            # Blend the overlay with the original frame
            alpha = 0.5  # Transparency factor
            current_frame = cv2.addWeighted(overlay_frame, alpha, current_frame, 1 - alpha, 0)

    # Display the processed frame
    cv2.imshow("Fire and Smoke Detection", current_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break  # Exit the loop if 'q' is pressed

    total_frames_processed += 1  # Increment the frame counter

# Wait for all email-sending threads to finish
for thread in email_threads:
    thread.join()  # Ensure all emails are sent before closing

# Release the video capture object and close the display window
video_capture.release()
cv2.destroyAllWindows()