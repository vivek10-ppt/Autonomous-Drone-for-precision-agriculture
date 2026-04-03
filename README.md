# Autonomous-Drone-for-precision-agriculture
Autonomous drone using Pixhawk (ArduCopter) controlled by Raspberry Pi via PyMAVLink. Generates lawnmower waypoints from KML, runs parallel ML detection, performs visual servoing for precise spray, avoids re-spraying nearby areas, and resumes mission automatically.


Here is a clean, ready-to-use README.md for your GitHub repo based on your actual code and setup:

🚁 Autonomous Precision Agriculture Drone

An autonomous drone system using Pixhawk (ArduCopter) and Raspberry Pi 5 (companion computer) controlled via PyMAVLink / DroneKit, capable of surveying farmland and performing ML-based precision spraying.

🔧 Features
KML-based survey area → lawnmower waypoint generation
Fully autonomous flight (no RC required)
Real-time ML detection (shared memory based)
Precision centering + descent + targeted spraying
Re-spray avoidance (radius-based filtering)
Automatic mission resume after spraying
RTL after mission completion
KML export of sprayed locations


🧠 System Architecture
Flight Controller: Pixhawk (ArduCopter firmware)
Companion Computer: Raspberry Pi 5 (8GB recommended)
Camera: Pi Camera Module 3 (recommended for best results)
Communication: MAVLink via USB / UART
ML Model: TensorFlow Lite (.tflite)
Detection Pipeline: Runs in parallel thread using shared memory


🔌 Hardware Setup
Required Components
Raspberry Pi 5 (8GB recommended)
Pixhawk Flight Controller
Pi Camera Module 3
Water pump + relay/driver
Power supply for Pi + FC
Connections
Pi ↔ Pixhawk: USB or UART
Camera → Pi CSI port
Water pump control:
GPIO 17 → relay / driver → pump


⚙️ Flight Controller Setup (IMPORTANT)

Ensure these are configured in Mission Planner:

MAVLink enabled on correct port
Correct baud rate (typically 115200)
Serial protocol set to MAVLink





1️⃣ Start ML Detection Thread (REQUIRED FIRST)
python Image_detection_thread.py
This initializes shared memory
Must run before main mission
2️⃣ Run Main Mission
python main.py --kml your_area.kml --connect /dev/ttyUSB0
Arguments:
--kml → input survey region
--connect → connection string (USB/UART)
--altitude → survey altitude
--spray-alt → spray altitude
🧪 Simulation Mode (No ML)
python mlsim.py
Runs only lawnmower survey
No spraying / detection
🧠 ML Model Setup

Place .tflite model in same folder as:

Image_detection_thread.py

Default expected name:

best_int8.tflite
💧 Spray System
Controlled via:
GPIO 17 (pump ON/OFF)


Logic implemented in:

activate_spray()
📍 Output

After mission:

Sprayed locations saved as:
waypoints/sprayed_locations.kml
Open in Google Earth to visualize spray points
⚠️ Important Notes
ML thread must run before main mission
Ensure GPS lock + home position set
Ensure safety switch / arming checks configured
Avoid running multiple MAVLink clients simultaneously
Test in simulation before real deployment
