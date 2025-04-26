import os
import sys
import time
import subprocess
import argparse
from multiprocessing import Process

# Replace this with your Graviton instance IP
SERVER_URL = "http://localhost:5000/detect"  # Default to localhost for testing

def run_camera(camera_id, server_url, video_file=None, reference_img="reference.jpg", use_webcam=False, webcam_id=None):
    """Run a camera simulation process"""
    cmd = [
        sys.executable,  # Use the current Python interpreter
        "camera.py",
        "--camera_id", camera_id,
        "--reference", reference_img,
        "--server", server_url
    ]
    
    if use_webcam:
        cmd.extend(["--use_webcam"])
        if webcam_id is not None:
            cmd.extend(["--webcam_id", str(webcam_id)])
        print(f"Starting camera {camera_id} with webcam {webcam_id if webcam_id is not None else 0}")
    else:
        cmd.extend(["--video", video_file])
        print(f"Starting camera {camera_id} with video {video_file}")
    
    # Start the process
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    return process

def main():
    parser = argparse.ArgumentParser(description='Run multiple camera simulations')
    parser.add_argument('--server_url', type=str, default=SERVER_URL,
                        help='URL of the server to send detections')
    parser.add_argument('--reference', type=str, default="reference.jpg",
                        help='Path to reference image for facial recognition')
    parser.add_argument('--use_webcam', action='store_true',
                        help='Use webcam instead of video files')
    parser.add_argument('--num_cameras', type=int, default=1,
                        help='Number of webcam simulations to run (only applies with --use_webcam)')
    args = parser.parse_args()
    
    # Check if reference image exists and warn if not (but don't exit, as webcam mode can capture one)
    if not os.path.exists(args.reference):
        print(f"Warning: Reference image not found: {args.reference}")
        if args.use_webcam:
            print("You will be prompted to capture a reference image from webcam")
        else:
            print("Please provide a valid reference image file or use --use_webcam option")
            sys.exit(1)
    
    # If using webcam mode
    if args.use_webcam:
        print(f"Starting {args.num_cameras} camera simulation(s) using webcam")
        
        # Check if we have enough webcams
        if args.num_cameras > 1:
            print("Warning: Running multiple webcam simulations may not work depending on available hardware")
            print("Trying to run multiple webcam instances...")
        
        # Start camera processes with webcam
        processes = []
        try:
            for i in range(args.num_cameras):
                camera_id = f"webcam_{i+1}"
                webcam_id = i  # Assign webcam IDs 0, 1, 2, etc.
                process = run_camera(
                    camera_id=camera_id,
                    server_url=args.server_url,
                    reference_img=args.reference,
                    use_webcam=True,
                    webcam_id=webcam_id
                )
                processes.append((camera_id, process))
        
        except Exception as e:
            print(f"Error starting webcam simulations: {str(e)}")
            sys.exit(1)
    
    # If using video files
    else:
        # Define video files for simulation
        # You may need to adjust these paths based on where you saved the videos
        video_files = [
            "View_001.avi",
            "View_002.avi",
            "View_003.avi"
        ]
        
        # Check if video files exist
        missing_videos = [v for v in video_files if not os.path.exists(v)]
        if missing_videos:
            print("Error: The following video files are missing:")
            for v in missing_videos:
                print(f"  - {v}")
            print("\nPlease download the video files as mentioned in the setup instructions,")
            print("or use --use_webcam option to use your webcam instead.")
            sys.exit(1)
        
        print(f"Starting camera simulations with video files. Server URL: {args.server_url}")
        
        # Start camera processes with video files
        processes = []
        try:
            for i, video_file in enumerate(video_files):
                camera_id = f"camera_{i+1}"
                process = run_camera(
                    camera_id=camera_id,
                    video_file=video_file,
                    reference_img=args.reference,
                    server_url=args.server_url
                )
                processes.append((camera_id, process))
        
        except Exception as e:
            print(f"Error starting video simulations: {str(e)}")
            sys.exit(1)
    
    print(f"Using reference image: {args.reference}")
    print(f"Server URL: {args.server_url}")
    print("All cameras started. Press Ctrl+C to stop all cameras.")
    
    # Monitor processes and display their output
    try:
        while True:
            for camera_id, process in processes:
                # Read output line by line
                if process.stdout:
                    line = process.stdout.readline()
                    if line:
                        print(f"[{camera_id}] {line.strip()}")
                
                # Check if process is still running
                if process.poll() is not None:
                    print(f"Camera {camera_id} exited with code {process.returncode}")
                    
                    # Restart the process if it exited
                    if args.use_webcam:
                        # For webcam, extract the webcam ID from the camera_id
                        idx = int(camera_id.split('_')[1]) - 1
                        webcam_id = idx
                        process = run_camera(
                            camera_id=camera_id, 
                            server_url=args.server_url,
                            reference_img=args.reference,
                            use_webcam=True,
                            webcam_id=webcam_id
                        )
                    else:
                        # For video files
                        idx = int(camera_id.split('_')[1]) - 1
                        video_file = video_files[idx]
                        process = run_camera(
                            camera_id=camera_id,
                            video_file=video_file,
                            reference_img=args.reference,
                            server_url=args.server_url
                        )
                    
                    # Update the process in the list
                    for i, (cam_id, _) in enumerate(processes):
                        if cam_id == camera_id:
                            processes[i] = (camera_id, process)
                            break
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nStopping all camera simulations...")
        for camera_id, process in processes:
            if process.poll() is None:  # If process is still running
                process.terminate()
                print(f"Terminated camera {camera_id}")
        
        # Wait for processes to terminate
        for camera_id, process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"Killed camera {camera_id} (did not terminate gracefully)")
        
        print("All cameras stopped.")

if __name__ == "__main__":
    main() 