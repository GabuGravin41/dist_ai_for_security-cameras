import os
import json
import datetime
from flask import Flask, request, jsonify

app = Flask(__name__)

# Path to store detection logs
DETECTION_LOG = "detections.json"

# Initialize log file if it doesn't exist
if not os.path.exists(DETECTION_LOG):
    with open(DETECTION_LOG, 'w') as f:
        json.dump([], f)

@app.route('/detect', methods=['POST'])
def receive_detection():
    """Endpoint to receive detection alerts from cameras"""
    if not request.json:
        return jsonify({"status": "error", "message": "No data provided"}), 400
    
    # Get detection data
    data = request.json
    
    # Add timestamp
    data['timestamp'] = datetime.datetime.now().isoformat()
    
    # Read existing detections
    with open(DETECTION_LOG, 'r') as f:
        detections = json.load(f)
    
    # Add new detection
    detections.append(data)
    
    # Save updated detections
    with open(DETECTION_LOG, 'w') as f:
        json.dump(detections, f, indent=2)
    
    return jsonify({"status": "success", "message": "Detection logged"}), 200

@app.route('/detections', methods=['GET'])
def get_detections():
    """Endpoint to view recent detections"""
    try:
        with open(DETECTION_LOG, 'r') as f:
            detections = json.load(f)
        
        # Return the 10 most recent detections
        recent_detections = detections[-10:] if detections else []
        
        # Basic HTML display
        html = "<html><head><title>Detection Alerts</title>"
        html += "<meta http-equiv='refresh' content='5'>"  # Auto-refresh every 5 seconds
        html += "<style>body{font-family:Arial;margin:20px} table{border-collapse:collapse;width:100%} th,td{border:1px solid #ddd;padding:8px;text-align:left}</style>"
        html += "</head><body><h1>Recent Detections</h1>"
        html += "<table><tr><th>Camera</th><th>Person</th><th>Confidence</th><th>Time</th></tr>"
        
        for det in reversed(recent_detections):
            html += f"<tr><td>{det.get('camera_id', 'Unknown')}</td><td>{det.get('person_id', 'Unknown')}</td>"
            html += f"<td>{det.get('confidence', 0):.2f}</td><td>{det.get('timestamp', '')}</td></tr>"
        
        html += "</table></body></html>"
        return html
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    print("Starting detection server on port 5000...")
    app.run(host='0.0.0.0', port=5000) 