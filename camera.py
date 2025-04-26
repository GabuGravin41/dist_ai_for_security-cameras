import cv2
import time
import requests
import argparse
import os
import numpy as np
from deepface import DeepFace

class CameraSimulation:
    def __init__(self, camera_id, video_path, reference_img_path, server_url, confidence_threshold=0.6, detection_interval=2, use_webcam=False, webcam_id=0):
        """
        Initialize a camera simulation
        
        Args:
            camera_id: Unique identifier for this camera
            video_path: Path to video file for simulation
            reference_img_path: Path to reference image for facial recognition
            server_url: URL of the server to send detections
            confidence_threshold: Minimum confidence for facial recognition (0-1)
            detection_interval: Minimum seconds between detection alerts
            use_webcam: Whether to use webcam instead of video file
            webcam_id: Webcam ID to use (default: 0)
        """
        self.camera_id = camera_id
        self.video_path = video_path
        self.reference_img_path = reference_img_path
        self.server_url = server_url
        self.confidence_threshold = confidence_threshold
        self.detection_interval = detection_interval
        self.last_detection_time = 0
        self.use_webcam = use_webcam
        self.webcam_id = webcam_id
        
        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Validate files exist if not using webcam
        if not self.use_webcam:
            self._validate_files()
        
        # Handle reference image
        self._setup_reference_image()
        
        print(f"Camera {self.camera_id} initialized with {'webcam' if self.use_webcam else 'video: ' + self.video_path}")
        
    def _validate_files(self):
        """Validate that required files exist"""
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        if not os.path.exists(self.reference_img_path):
            raise FileNotFoundError(f"Reference image not found: {self.reference_img_path}")
    
    def _setup_reference_image(self):
        """
        Setup reference image for facial recognition
        """
        # Check if reference image exists
        if os.path.exists(self.reference_img_path):
            # Load reference image for facial recognition
            self.reference_img = cv2.imread(self.reference_img_path)
            if self.reference_img is None:
                if self.use_webcam:
                    self._capture_reference_image()
                else:
                    raise FileNotFoundError(f"Could not load reference image: {self.reference_img_path}")
        else:
            if self.use_webcam:
                self._capture_reference_image()
            else:
                raise FileNotFoundError(f"Reference image not found: {self.reference_img_path}")
    
    def _capture_reference_image(self):
        """Capture reference image from webcam"""
        print(f"Reference image not found. Capturing from webcam...")
        
        # Initialize camera
        cap = cv2.VideoCapture(self.webcam_id)
        
        if not cap.isOpened():
            raise Exception("Could not open webcam")
        
        print("Press 'c' to capture reference image when ready")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                continue
                
            # Display frame
            cv2.imshow("Capture Reference Image", frame)
            
            # Check for keypress
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                # Save frame as reference image
                cv2.imwrite(self.reference_img_path, frame)
                self.reference_img = frame
                print(f"Reference image captured and saved to {self.reference_img_path}")
                break
            elif key == ord('q'):
                raise Exception("Reference image capture cancelled")
        
        # Clean up
        cap.release()
        cv2.destroyWindow("Capture Reference Image")
    
    def detect_faces(self, frame):
        """
        Detect and recognize faces in a frame
        
        Returns:
            List of (face_coords, confidence) tuples for recognized faces
        """
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with OpenCV (faster initial screening)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        
        results = []
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            
            try:
                # Use DeepFace to verify if this face matches the reference
                verification = DeepFace.verify(
                    img1_path=face_img,
                    img2_path=self.reference_img_path,
                    enforce_detection=False,
                    model_name="VGG-Face"
                )
                
                # Get the verification confidence and check threshold
                confidence = verification["distance"]
                recognized = verification["verified"]
                
                if recognized and confidence >= self.confidence_threshold:
                    results.append(((x, y, w, h), confidence))
            
            except Exception as e:
                # If face verification fails, just continue
                continue
        
        return results
    
    def send_alert(self, confidence):
        """Send detection alert to server"""
        current_time = time.time()
        
        # Only send alerts at specified intervals
        if current_time - self.last_detection_time < self.detection_interval:
            return
        
        # Update last detection time
        self.last_detection_time = current_time
        
        # Prepare data for sending
        data = {
            "camera_id": self.camera_id,
            "person_id": "target_person",  # In a real system, this would be the identified person
            "confidence": float(confidence),
            "location": "simulated_location"
        }
        
        try:
            # Send data to server
            response = requests.post(self.server_url, json=data)
            if response.status_code == 200:
                print(f"Camera {self.camera_id}: Alert sent successfully")
            else:
                print(f"Camera {self.camera_id}: Failed to send alert - {response.status_code}")
        except Exception as e:
            print(f"Camera {self.camera_id}: Error sending alert - {str(e)}")
    
    def run(self):
        """Run the camera simulation"""
        print(f"Camera {self.camera_id}: Starting video processing")
        
        # Open video capture (webcam or file)
        if self.use_webcam:
            cap = cv2.VideoCapture(self.webcam_id)
            print(f"Camera {self.camera_id}: Using webcam #{self.webcam_id}")
        else:
            cap = cv2.VideoCapture(self.video_path)
            print(f"Camera {self.camera_id}: Using video file {self.video_path}")
        
        if not cap.isOpened():
            raise Exception(f"Could not open {'webcam' if self.use_webcam else 'video file: ' + self.video_path}")
        
        frame_count = 0
        processing_interval = 5  # Process every 5th frame to reduce CPU usage
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                if self.use_webcam:
                    print(f"Camera {self.camera_id}: Lost webcam feed, retrying...")
                    time.sleep(1)
                    continue
                else:
                    print(f"Camera {self.camera_id}: End of video reached, restarting...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop back to beginning
                    continue
            
            frame_count += 1
            
            # Only process every few frames to save CPU
            if frame_count % processing_interval == 0:
                # Detect and recognize faces
                detections = self.detect_faces(frame)
                
                # If faces detected, send alerts
                for (face_coords, confidence) in detections:
                    (x, y, w, h) = face_coords
                    
                    # Draw rectangle around detected face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Add text with confidence score
                    cv2.putText(frame, f"Match: {confidence:.2f}", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Send alert to server
                    self.send_alert(confidence)
                
                # Show frame
                cv2.imshow(f'Camera {self.camera_id}', frame)
            
            # Break loop on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Slow down simulation to match real-time speed (approximate)
            if not self.use_webcam:
                time.sleep(0.03)  # ~30fps for video files
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print(f"Camera {self.camera_id}: Simulation complete")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run a simulated CCTV camera with facial recognition')
    parser.add_argument('--camera_id', type=str, required=True, help='Unique identifier for this camera')
    parser.add_argument('--video', type=str, help='Path to video file (required if not using webcam)')
    parser.add_argument('--reference', type=str, default="reference.jpg", help='Path to reference image')
    parser.add_argument('--server', type=str, required=True, help='URL of the server to send detections')
    parser.add_argument('--threshold', type=float, default=0.6, help='Confidence threshold (0-1)')
    parser.add_argument('--interval', type=int, default=2, help='Seconds between detection alerts')
    parser.add_argument('--use_webcam', action='store_true', help='Use webcam instead of video file')
    parser.add_argument('--webcam_id', type=int, default=0, help='Webcam ID to use (default: 0)')
    
    args = parser.parse_args()
    
    # Validate args
    if not args.use_webcam and not args.video:
        parser.error("Either --video or --use_webcam must be specified")
    
    # Create and run camera simulation
    camera = CameraSimulation(
        camera_id=args.camera_id,
        video_path=args.video if not args.use_webcam else None,
        reference_img_path=args.reference,
        server_url=args.server,
        confidence_threshold=args.threshold,
        detection_interval=args.interval,
        use_webcam=args.use_webcam,
        webcam_id=args.webcam_id
    )
    
    camera.run() 