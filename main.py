from pykml import parser
from shapely.geometry import Polygon, LineString, Point
from dronekit import connect, VehicleMode, LocationGlobalRelative
import json
import os
import math
import time
import argparse
from pymavlink import mavutil
MIN_ALTITUDE = 1.5  # meters, absolute safety floor
SPRAY_SERVO = 9      # AUX1 ? SERVO9
OPEN_PWM = 1900
CLOSE_PWM = 1100

# ===== SHARED MEMORY INTERFACE (To be implemented with actual shared memory) =====
import struct
from multiprocessing import shared_memory
#from lidar1 import LidarThread ,obstacle_flag, paused_flag

def export_sprayed_locations_kml(sprayed_locations, out_path="waypoints/sprayed_locations.kml"):
    if not sprayed_locations:
        print("ℹ No sprayed locations to export.")
        return None

    folder = os.path.dirname(out_path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        '<Document>',
        '<name>Sprayed Locations</name>',
        """
        <Style id="sprayedStyle">
          <IconStyle>
            <scale>1.1</scale>
            <Icon>
              <href>http://maps.google.com/mapfiles/kml/pushpin/ylw-pushpin.png</href>
            </Icon>
          </IconStyle>
        </Style>
        """.strip()
    ]

    ok = 0
    for i, loc in enumerate(sprayed_locations, start=1):
        try:
            lat = float(loc.lat)
            lon = float(loc.lon)
            alt = float(getattr(loc, "alt", 0.0))

            # basic sanity check
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                raise ValueError(f"Invalid lat/lon: {lat},{lon}")

            lines += [
                "<Placemark>",
                f"<name>Sprayed {i}</name>",
                "<styleUrl>#sprayedStyle</styleUrl>",
                "<Point>",
                "<altitudeMode>clampToGround</altitudeMode>",
                f"<coordinates>{lon},{lat},{alt}</coordinates>",
                "</Point>",
                "</Placemark>",
            ]
            ok += 1

        except Exception as e:
            print(f"⚠ Skipping sprayed point #{i} due to error: {e}")

    lines += ["</Document>", "</kml>"]

    if ok == 0:
        print("❌ No valid sprayed points to export.")
        return None

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return out_path
class MLDetectionInterface:
    """
    Interface for reading ML model outputs from shared memory
    Shared memory layout: (detected:int, x:int, y:int)
    """

    SHM_NAME = "ml_detection_shm"
    STRUCT_FORMAT = "iii"   # detected, x_offset, y_offset
    STRUCT_SIZE = struct.calcsize(STRUCT_FORMAT)

    def __init__(self):
        self.last_seen_time = 0

        # Attach to shared memory created by writer process
        self.shm = shared_memory.SharedMemory(name=self.SHM_NAME)
        self.buf = self.shm.buf

    def read_detection(self):
        """
        Read detection data from shared memory

        Returns:
            (detected: bool, x_offset: int, y_offset: int)
        """
        detected, x_offset, y_offset = struct.unpack(
            self.STRUCT_FORMAT,
            self.buf[:self.STRUCT_SIZE]
        )

        # Freshness handling (optional but good)
        if detected:
            self.last_seen_time = time.time()

        # Invalidate if stale (>2 seconds)
        if time.time() - self.last_seen_time > 2:
            return False, 0, 0

        return bool(detected), x_offset, y_offset

    def clear_detection(self):
        """
        Clear detection (writer will overwrite anyway)
        This is kept only for API compatibility
        """
        pass


# Create global instance
ml_detector = MLDetectionInterface()

# ===== WAYPOINT GENERATOR =====
class WaypointGenerator:
    def __init__(self, drone1_kml_path, line_spacing_meters=5):
        self.drone1_kml_path = drone1_kml_path
        self.line_spacing_meters = line_spacing_meters
        self.survey_polygon = None
        self.load_drone1_area()
    
    def load_drone1_area(self):
        """Load the drone1 area KML file"""
        with open(self.drone1_kml_path, 'r') as f:
            doc = parser.parse(f)
        
        root = doc.getroot()
        placemark = root.Document.Placemark
        coords_str = str(placemark.Polygon.outerBoundaryIs.LinearRing.coordinates)
        
        coords = []
        for coord in coords_str.strip().split():
            parts = coord.split(',')
            lon, lat = float(parts[0]), float(parts[1])
            coords.append((lon, lat))
        
        self.survey_polygon = Polygon(coords)
        print(f"Loaded survey area: {len(coords)} vertices")
    
    def meters_to_degrees(self, meters):
        """Convert meters to approximate degrees"""
        return meters / 111320.0
    
    def generate_lawnmower_waypoints(self):
        """Generate lawnmower pattern waypoints"""
        min_lon, min_lat, max_lon, max_lat = self.survey_polygon.bounds
        line_spacing_deg = self.meters_to_degrees(self.line_spacing_meters)
        
        width = max_lon - min_lon
        height = max_lat - min_lat
        
        if width > height:
            waypoints = self._generate_east_west_lines(min_lat, max_lat, min_lon, max_lon, line_spacing_deg)
        else:
            waypoints = self._generate_north_south_lines(min_lat, max_lat, min_lon, max_lon, line_spacing_deg)
        
        print(f"Generated {len(waypoints)} waypoints")
        return waypoints
    
    def _generate_east_west_lines(self, min_lat, max_lat, min_lon, max_lon, line_spacing):
        """Generate east-west survey lines"""
        waypoints = []
        current_lat = max_lat
        line_number = 0
        
        while current_lat >= min_lat:
            survey_line = LineString([(min_lon, current_lat), (max_lon, current_lat)])
            intersection = self.survey_polygon.intersection(survey_line)
            
            if not intersection.is_empty and intersection.geom_type == 'LineString':
                coords = list(intersection.coords)
                if len(coords) >= 2:
                    if line_number % 2 == 0:
                        waypoints.append(Point(coords[0]))
                        waypoints.append(Point(coords[-1]))
                    else:
                        waypoints.append(Point(coords[-1]))
                        waypoints.append(Point(coords[0]))
                    line_number += 1
            
            current_lat -= line_spacing
        
        return waypoints
    
    def _generate_north_south_lines(self, min_lat, max_lat, min_lon, max_lon, line_spacing):
        """Generate north-south survey lines"""
        waypoints = []
        current_lon = max_lon
        line_number = 0
        
        while current_lon >= min_lon:
            survey_line = LineString([(current_lon, min_lat), (current_lon, max_lat)])
            intersection = self.survey_polygon.intersection(survey_line)
            
            if not intersection.is_empty and intersection.geom_type == 'LineString':
                coords = list(intersection.coords)
                if len(coords) >= 2:
                    if line_number % 2 == 0:
                        waypoints.append(Point(coords[0]))
                        waypoints.append(Point(coords[-1]))
                    else:
                        waypoints.append(Point(coords[-1]))
                        waypoints.append(Point(coords[0]))
                    line_number += 1
            
            current_lon -= line_spacing
        
        return waypoints
    
    def save_waypoints(self, waypoints, output_dir="waypoints"):
        """Save waypoints to JSON file"""
        os.makedirs(output_dir, exist_ok=True)
        
        waypoint_list = []
        for i, point in enumerate(waypoints):
            waypoint_list.append({
                'id': i,
                'latitude': point.y,
                'longitude': point.x,
                'altitude': 8
            })
        
        base_name = os.path.splitext(os.path.basename(self.drone1_kml_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}_waypoints.json")
        
        with open(output_file, 'w') as f:
            json.dump(waypoint_list, f, indent=2)
        
        print(f"Saved waypoints to: {output_file}")
        return output_file, waypoint_list

# ===== AUTONOMOUS SPRAY MISSION =====
class AutonomousSprayMission:
    def __init__(self, connection_string='/dev/serial0'):
        self.connection_string = connection_string
        self.vehicle = None
        #self.lidar_thread = None
        self.sprayed_locations = []
        self.spray_cooldown = 10  # Seconds to ignore detections after spraying
        self.last_spray_time = 0
        self.spray_altitude = 3  # Meters above ground for spraying

    def set_servo(self, servo_number, pwm):
        """
        servo_number: SERVO output number (e.g., 9 for AUX1)
        pwm: 1000 2000
        """
        msg = self.vehicle.message_factory.command_long_encode(
            0, 0,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
            0,
            servo_number,
            pwm,
            0, 0, 0, 0, 0
        )
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()

    def go_to_altitude(self, target_alt, tolerance=0.3, timeout=20):
        """
        Safely climb/descend to a specific altitude using feedback
        """
        print(f"Going to altitude: {target_alt:.1f} m")

        target = LocationGlobalRelative(
            self.vehicle.location.global_relative_frame.lat,
            self.vehicle.location.global_relative_frame.lon,
            target_alt
        )

        self.vehicle.simple_goto(target)

        start = time.time()
        while True:
            if self.vehicle.mode.name != "GUIDED":
                print("Mode changed, aborting altitude change")
                return False

            current_alt = self.vehicle.location.global_relative_frame.alt
            error = abs(current_alt - target_alt)

            print(f" Altitude: {current_alt:.2f} m", end="\r")

            if error < tolerance:
                print("\nReached target altitude")
                return True

            if time.time() - start > timeout:
                print("\nAltitude change timeout")
                return False

            time.sleep(0.3)
    def is_near_sprayed_location(self, location, radius=3.0):
        """
        Check if current location is near any previously sprayed location
        """
        for loc in self.sprayed_locations:
            if self.get_distance_metres(location, loc) < radius:
                return True
        return False

    # ===== BASIC DRONE CONTROL =====
    def connect_vehicle(self):
        """Connect to vehicle"""
        print(f"Connecting to {self.connection_string}...")
        self.vehicle = connect(self.connection_string,baud=115200,wait_ready=True,timeout=60)

        print("âœ“ Connected to vehicle!")
        #self.lidar_thread = LidarThread(self.vehicle, port="/dev/ttyUSB0", baud=115200)
        #self.lidar_thread.start()
        #print("??? LiDAR ? FC streaming ENABLED (10 Hz)")
        
        print("------------------------------------------")
        print(f"Mode: {self.vehicle.mode.name}")
        print(f"GPS: {self.vehicle.gps_0.fix_type}")
        print(f"Home: {self.vehicle.home_location}")
        print("------------------------------------------")
    
    def arm_and_takeoff(self, target_altitude):
        """Arm and takeoff"""
        print("Basic pre-arm checks")
        start = time.time()
        while not self.vehicle.is_armable:
            print("not armable")
            if time.time() - start > 60:
                raise RuntimeError("Vehicle not armable (GPS / EKF / Safety switch)")
            time.sleep(1)
        while not self.vehicle.home_location:
           print("Waiting for HOME position...")
           time.sleep(1)

        print("Arming motors")
        self.vehicle.mode = VehicleMode("GUIDED")
        self.vehicle.armed = True

        while not self.vehicle.armed:
            print(" Waiting for arming...")
            time.sleep(1)

        print("Taking off!")
        self.vehicle.simple_takeoff(target_altitude)

        while True:
            altitude = self.vehicle.location.global_relative_frame.alt
            print(f" Altitude: {altitude:.2f} meters")
            if altitude >= target_altitude * 0.95:
                print("Reached target altitude")
                break
            time.sleep(1)
    
    def get_heading(self):
        """Get current heading in degrees (0-360)"""
        return math.degrees(self.vehicle.attitude.yaw) % 360
    def lock_current_yaw(self):
        """
        Lock the drone's yaw to its current heading
        """
        heading = self.get_heading()

        msg = self.vehicle.message_factory.command_long_encode(
            0, 0,
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            0,
            heading,   # target heading (deg)
            0,         # yaw speed (0 = default)
            1,         # direction (ignored when speed=0)
            0,         # absolute heading
            0, 0, 0
        )

        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()

    def stop_drone(self):
        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0, 0, 0,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            0b0000111111000111,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0, 0, 0
        )
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()


    def move_xy_velocity(self, vx, vy, duration=1.0):
        """
        Move drone using body-frame velocities (no yaw change)
        vx: forward (+) / backward (-) in m/s
        vy: right (+) / left (-) in m/s
        """
        if self.vehicle.mode.name != "GUIDED":
            print("Mode changed, aborting velocity move")
            return False
        self.lock_current_yaw()
        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0, 0, 0,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            0b0000111111000111,
            0, 0, 0,
            vx, vy, 0,
            0, 0, 0, 0, 0
        )

        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()

        start = time.time()
        while time.time() - start < duration:
            if self.vehicle.mode.name != "GUIDED":
                print("Mode changed, stopping velocity")
                break
            time.sleep(0.1)

        # Stop
        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0, 0, 0,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            0b0000111111000111,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0, 0, 0
        )
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()

        return True

    def condition_yaw(self, heading_degrees, relative=False):
        """Set yaw to specific heading"""
        heading_degrees = heading_degrees 
        if relative:
            current_heading = self.get_heading()
            target_heading = (current_heading + heading_degrees) % 360
        else:
            target_heading = heading_degrees % 360
        
        print(f"Yawing to: {target_heading:.1f}Â° (current: {self.get_heading():.1f}Â°)")
        
        msg = self.vehicle.message_factory.command_long_encode(
            0, 0,
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            0,
            target_heading,
            45,
            -1 if heading_degrees < 0 else 1,
            0,
            0, 0, 0
        )
        self.vehicle.send_mavlink(msg)
        time.sleep(1)  # Allow yaw to start
        return target_heading
    
    def move_forward_precise(self, distance_meters=1, speed=0.5):
        """Move precise distance forward relative to current heading"""
        print(f"Moving {distance_meters}m forward at {speed} m/s")
        distance_meters = min(distance_meters, 2.0)

        move_time = distance_meters / speed
        
        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0, 0, 0,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            0b0000111111000111,
            0, 0, 0,
            speed, 0, 0,
            0, 0, 0, 0, 0
        )
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()
        
        print(f"Moving for {move_time:.2f} seconds...")
        time.sleep(move_time)
        
        # Stop
        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0, 0, 0,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            0b0000111111000111,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0, 0, 0
        )
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()
        
        print("âœ“ Movement complete")
        time.sleep(1)  # Stabilization
    
    def move_vertical(self, distance_meters=1, speed=0.5, direction='down'):
        """Move vertically up or down"""
        if direction == 'down':
            vz = speed
            print(f"Descending {distance_meters}m at {speed} m/s")
        else:
            vz = -speed
            print(f"Ascending {distance_meters}m at {speed} m/s")
        
        move_time = distance_meters / speed
        
        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0, 0, 0,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            0b0000111111000111,
            0, 0, 0,
            0, 0, vz,
            0, 0, 0, 0, 0
        )
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()
        
        time.sleep(move_time)
        
        # Stop
        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0, 0, 0,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            0b0000111111000111,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0, 0, 0
        )
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()
        
        print(f"âœ“ Vertical movement complete")
        time.sleep(1)  # Stabilization
    def move_vertical_by_1m(self, direction='down'):
        """
        Safely move drone up or down by 1 meter
        Uses altitude target instead of time-based velocity
        """

        current_alt = self.vehicle.location.global_relative_frame.alt

        if direction == 'down':
            target_alt = current_alt - 1.0
            print(f"Descending to {target_alt:.2f} m")
        else:
            target_alt = current_alt + 1.0
            print(f"Ascending to {target_alt:.2f} m")

        # Safety floor
        if target_alt < MIN_ALTITUDE:
            print("Reached minimum safe altitude, aborting descent")
            return False

        target = LocationGlobalRelative(
            self.vehicle.location.global_relative_frame.lat,
            self.vehicle.location.global_relative_frame.lon,
            target_alt
        )

        self.vehicle.simple_goto(target)

        # Wait until altitude reached
        while True:
            if self.vehicle.mode.name != "GUIDED":
                print("Mode changed, aborting vertical movement")
                return False

            alt = self.vehicle.location.global_relative_frame.alt
            error = abs(alt - target_alt)

            print(f" Altitude: {alt:.2f} m", end="\r")

            if error < 0.15:
                print("\nReached target altitude")
                return True

            time.sleep(0.3)

    def goto_waypoint(self, waypoint, groundspeed=5):
        """Go to specific waypoint"""
        target = LocationGlobalRelative(
            waypoint['latitude'],
            waypoint['longitude'],
            waypoint.get('altitude', 8)
        )
        
        print(f"Flying to WP: Lat={target.lat:.6f}, Lon={target.lon:.6f}, Alt={target.alt}m")
        self.vehicle.simple_goto(target, groundspeed=groundspeed)
        return target
    
    def get_distance_metres(self, location1, location2):
        """Calculate distance between two positions in meters (supports dict or Location objects)"""
        # unify dict ? object values
        if isinstance(location2, dict):
            lat2 = location2["latitude"]
            lon2 = location2["longitude"]
        else:
            lat2 = location2.lat
            lon2 = location2.lon 

        lat1 = location1.lat
        lon1 = location1.lon

        dlat = lat2 - lat1
        dlong = lon2 - lon1
        return math.sqrt((dlat * 111320)**2 + (dlong * 111320 * math.cos(math.radians(lat1)))**2)

    
    def fly_to_waypoint_with_detection(self,target,acceptance_radius=2,timeout=120):
        start = time.time()

        while time.time() - start < timeout:
            if self.vehicle.mode.name != "GUIDED":
                print("Mode changed, aborting waypoint flight")
                return False

            current = self.vehicle.location.global_relative_frame
            distance = self.get_distance_metres(current, target)

            print(f"Distance to WP: {distance:.2f} m", end="\r")
            '''if obstacle_flag.value:
                self.stop_drone()
                paused_flag.value = True

                while obstacle_flag.value:
                    time.sleep(0.3)

                paused_flag.value = False
                return "abort"'''
            #  CONTINUOUS DETECTION
            #  CONTINUOUS DETECTION WITH TEMPORAL CONFIRMATION
            detected, x_offset, y_offset = ml_detector.read_detection()

            if time.time() - self.last_spray_time < self.spray_cooldown:
                print("Cooldown active  ignoring detection")
                detected = False

            if detected:
                current_location = self.vehicle.location.global_frame

                if self.is_near_sprayed_location(current_location):
                    print("Already sprayed nearby location, ignoring detection")
                    ml_detector.clear_detection()
                    return "abort"   # continue waypoint flight
                print(" Detection seen stopping drone")
                self.stop_drone()

                # allow motion + camera to settle
                time.sleep(0.5)

                # re-check detection after stopping
                detected2, x2, y2 = ml_detector.read_detection()

                if detected2:
                    current_location = self.vehicle.location.global_frame

                    if self.is_near_sprayed_location(current_location):
                        print("Already sprayed nearby location, ignoring detection")
                        ml_detector.clear_detection()
                        return "sprayed"   # continue waypoint flight

                    print("? Detection confirmed after stop -tarting spray sequence")
                    return self.execute_spray_sequence()

                else:
                    print("False detection -resuming survey")
                    return "abort"


            if distance <= acceptance_radius:
                print("\n Waypoint reached")
                return "arrived"

            time.sleep(0.3)

        print("\n Waypoint timeout")
        return "timeout"

    # ===== CENTERING ALGORITHM (Real-world ready) =====
    def center_on_object(self, initial_x, initial_y, max_iterations=15):
        """
        Center drone on detected object using real-time pixel offsets
        
        Args:
            initial_x, initial_y: Initial pixel offsets
            max_iterations: Maximum centering attempts
        
        Returns:
            bool: True if centered, False if failed
        """
        print(f"\n" + "="*60)
        print(f"CENTERING ALGORITHM STARTED")
        print(f"Initial offsets: x={initial_x}, y={initial_y} pixels")
        print("="*60)
        
        iteration = 0
        current_altitude = self.vehicle.location.global_relative_frame.alt
        
        while iteration < max_iterations:
            iteration += 1
            
            # Get CURRENT offsets from shared memory (ML model)
            detected, x_offset, y_offset = ml_detector.read_detection()
            
            if not detected:
                print("Object lost during centering")
                return False
            
            # Calculate pixel distance
            pixel_distance = math.sqrt(x_offset**2 + y_offset**2)
            print(f"\nIteration {iteration}: x={x_offset:.0f}, y={y_offset:.0f}, dist={pixel_distance:.1f}px")
            
            # Check if centered (within 20 pixels)
            if pixel_distance < 100:
                print(f" Centered! Offset distance: {pixel_distance:.1f} pixels")
                return True
            
          # Convert pixels  meters (your existing logic)
            conversion_factor = 0.005 * (current_altitude / 10)
            dx_m = x_offset * conversion_factor
            dy_m = y_offset * conversion_factor

            # Convert meters  velocity (clamped)
            MAX_VEL = 0.3  # m/s (safe)
            vx = max(-MAX_VEL, min(-dy_m, MAX_VEL))
            vy = max(-MAX_VEL, min(dx_m, MAX_VEL))

            print(f"Velocity command: vx={vx:.2f}, vy={vy:.2f}")

            self.move_xy_velocity(vx, vy, duration=1.0)

            
            # Wait for camera to update
            print("Waiting for camera feedback...")
            time.sleep(2)  # Allow ML model to process new frame
        
        print(f"âœ— Centering failed after {max_iterations} iterations")
        return False
    
    # ===== SPRAYING SEQUENCE =====
    def execute_spray_sequence(self):
        """
        Complete spray sequence: descend, center, spray, ascend
        Uses real-time offsets from shared memory
        """
        print("\n" + "="*60)
        print("SPRAY SEQUENCE ACTIVATED")
        print("="*60)
        
        # Store starting altitude and location
        start_altitude = self.vehicle.location.global_relative_frame.alt
        start_location = self.vehicle.location.global_frame
        
        print(f"Starting altitude: {start_altitude:.1f}m")
        print(f"Starting location: {start_location.lat:.6f}, {start_location.lon:.6f}")
        
        # Get initial detection
        detected, x_offset, y_offset = ml_detector.read_detection()
        if not detected:
            print("No object detected, aborting spray sequence")
            return "abort"
        
        # Phase 1: Center at current altitude
        print("\nPhase 1: Centering at current altitude...")
        if not self.center_on_object(x_offset, y_offset):
            print("Failed to center at survey altitude")
            return "abort"
        
        # Phase 2: Descend in steps, re-centering at each level
        print("\nPhase 2: Stepwise descent with re-centering")
        current_altitude = self.vehicle.location.global_relative_frame.alt

        while current_altitude > self.spray_altitude:
            print(f"\nDescending from {current_altitude:.1f} m")

            success = self.move_vertical_by_1m(direction='down')
            if not success:
                print("Failed during controlled descent, aborting spray")
                return "abort"

            current_altitude = self.vehicle.location.global_relative_frame.alt
            print(f"At {current_altitude:.1f}m - Re-centering...")

            time.sleep(0.8)  # camera stabilization only (not motion!)

            detected, x_offset, y_offset = ml_detector.read_detection()
            if not detected:
                print("Object lost during descent, ascending back")
                self.move_vertical_by_1m(direction='up')
                return "abort"

            if not self.center_on_object(x_offset, y_offset, max_iterations=6):
                print("Centering failed during descent, aborting")
                self.go_to_altitude(start_altitude)
                return "abort"

                
        # Phase 3: Final position check at spray altitude
        
        print(f"\nPhase 3: At spray altitude ({self.spray_altitude}m)")

        detected, x_offset, y_offset = ml_detector.read_detection()

        if not detected:
            print("Object lost at spray altitude, aborting and climbing back")
            self.go_to_altitude(start_altitude)
            return "abort"

        offset = math.sqrt(x_offset**2 + y_offset**2)

        if offset < 160:
            print("Ready to spray!")
        else:
            print(f"Offset {offset:.1f}px too large, re-centering at spray altitude")

            if not self.center_on_object(x_offset, y_offset, max_iterations=5):
                print("Failed to re-center at spray altitude, aborting")
                self.go_to_altitude(start_altitude)
                return "abort"

            print(" Re-centered successfully, ready to spray")

        
        # TRIGGER SPRAY
        print("\n" + "="*60)
        print("TRIGGERING SPRAY SYSTEM")
        print("="*60)
        
        spray_success = self.activate_spray()
        
        if spray_success:
            # Mark location as sprayed
            current_location = self.vehicle.location.global_frame
            self.sprayed_locations.append(LocationGlobalRelative(current_location.lat,current_location.lon, self.spray_altitude))

            
            print(f" Location sprayed: {current_location.lat:.6f}, {current_location.lon:.6f}")
            self.last_spray_time = time.time()
            
            # Clear detection in shared memory
            ml_detector.clear_detection()
        
        # Phase 4: Ascend back to survey altitude
        print(f"\nPhase 4: Returning to survey altitude ({start_altitude:.1f}m)")
        if not self.go_to_altitude(start_altitude):
            print("Failed to return to survey altitude, aborting mission")
            return "abort"

        
        print("\n" + "="*60)
        print("SPRAY SEQUENCE COMPLETED!")
        print("="*60)
        
        if spray_success:
            return "sprayed"
        else:
            return "abort"
    
    def activate_spray(self):
        """Activate spray system via GPIO (lgpio)"""

        CHIP = 0
        SPRAY_PIN = 17
        SPRAY_DURATION = 10

        try:
            import lgpio

            # Open GPIO chip
            h = lgpio.gpiochip_open(CHIP)

            # Claim pin as output
            lgpio.gpio_claim_output(h, SPRAY_PIN)
            self.move_forward_precise(0.25, speed=0.25)
            self.set_servo(SPRAY_SERVO, OPEN_PWM)
            print("Activating spray...")
            lgpio.gpio_write(h, SPRAY_PIN, 1)

            time.sleep(SPRAY_DURATION)

            lgpio.gpio_write(h, SPRAY_PIN, 0)
            self.set_servo(SPRAY_SERVO, CLOSE_PWM)
            print("? Spray completed")

            lgpio.gpiochip_close(h)
            return True

        except ImportError:
            # Simulation mode if lgpio not installed / not on Pi
            print("(GPIO not available - simulation mode)")
            print("Spray activated for 3 seconds...")
            time.sleep(3)
            print("? Spray completed (simulated)")
            return True

        except Exception as e:
            print(f"? Spray activation failed: {e}")
            try:
                # Best effort: close handle if it exists
                if 'h' in locals():
                    lgpio.gpiochip_close(h)
            except:
                pass
            return False

        
    # ===== MAIN MISSION EXECUTION =====
    def execute_survey_with_spray(self, waypoints, survey_altitude=8, groundspeed=2):
        print(f"\nStarting autonomous survey with {len(waypoints)} waypoints")
        print(f"Survey altitude: {survey_altitude}m, Speed: {groundspeed} m/s")

        for idx, wp in enumerate(waypoints):
            print(f"\n" + "="*60)
            print(f"WAYPOINT {idx+1}/{len(waypoints)}")
            print("="*60)

            # LOOP UNTIL waypoint physically reached
            while True:
                target = self.goto_waypoint(wp,groundspeed)
                result = self.fly_to_waypoint_with_detection(wp)

                # 1NORMAL ARRIVAL
                if result == "arrived":
                    print(f"Waypoint {idx+1} reached")
                    break  # move to next waypoint

                # 2 SPRAY OCCURRED
                elif result == "sprayed":
                    print(" Spray complete â€” returning to survey altitude...")
                    self.go_to_altitude(survey_altitude)  # climb back
                    time.sleep(1)
                    print("Resuming toward SAME waypoint...")
                    continue  # continue flying toward same waypoint

                # 3 ABORT (pilot override / failsafe / lost mode)
                elif result == "abort":
                    print("aborting spray")
                    continue
                elif result == "timeout":
                    print(" WP timeout")
                    break

            # OPTIONAL small wait between waypoints
            time.sleep(1)

        print("\n" + "="*60)
        print("SURVEY MISSION COMPLETE!")
        print(f"Total sprayed targets: {len(self.sprayed_locations)}")
        print("="*60)

    def return_to_launch(self):
        """Return to launch and land"""
        print("Returning to launch...")
        self.vehicle.mode = VehicleMode("RTL")
        
        while self.vehicle.armed:
            alt = self.vehicle.location.global_relative_frame.alt
            print(f"Altitude: {alt:.2f}m")
            time.sleep(2)
            
        print(" Landed successfully!")
    
    def close_connection(self):
        '''if self.lidar_thread:
            print("Stopping LiDAR thread...")
            self.lidar_thread.stop()'''
        """Close vehicle connection"""
        if self.vehicle:
            self.vehicle.close()
            print("Connection closed")

# ===== MAIN EXECUTION =====
def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Autonomous Drone Spray Mission')
    parser.add_argument('--kml', required=True, help='Path to KML file with survey area')
    parser.add_argument('--connect', default='/dev/ttyUSB0', help='Drone connection string')
    parser.add_argument('--spacing', type=float, default=3.75, help='Line spacing in meters')
    parser.add_argument('--altitude', type=float, default=6, help='Survey altitude in meters')
    parser.add_argument('--speed', type=float, default=2, help='Flight speed in m/s')
    parser.add_argument('--spray-alt', type=float, default=2, help='Spray altitude in meters')
    
    args = parser.parse_args()
    
    print("="*70)
    print("AUTONOMOUS DRONE SPRAY MISSION")
    print("="*70)
    print(f"Using shared memory interface for ML detections")
    print(f"Make sure ML model is running and updating shared memory")
    print("="*70)
    
    # Step 1: Generate waypoints
    print("\n1. GENERATING WAYPOINTS FROM KML")
    print("-"*40)
    
    if not os.path.exists(args.kml):
        print(f" KML file not found: {args.kml}")
        return
    
    try:
        generator = WaypointGenerator(args.kml, args.spacing)
        waypoints_geometric = generator.generate_lawnmower_waypoints()
        waypoints_file, waypoints_list = generator.save_waypoints(waypoints_geometric)
        print(f" Waypoints saved to: {waypoints_file}")
        
    except Exception as e:
        print(f" Error generating waypoints: {e}")
        return
    
    # Step 2: Execute mission
    print("\n2. EXECUTING AUTONOMOUS SPRAY MISSION")
    print("-"*40)
    
    mission = AutonomousSprayMission(args.connect)
    
    try:
        mission.connect_vehicle()
        mission.arm_and_takeoff(args.altitude)
        
        print("Hovering for stabilization...")
        time.sleep(5)
        
        mission.spray_altitude = args.spray_alt
        mission.execute_survey_with_spray(waypoints_list, args.altitude, args.speed)
        mission.return_to_launch()
        
        print("\n" + "="*70)
        print("MISSION COMPLETED SUCCESSFULLY! ")
        print(f"Total crops sprayed: {len(mission.sprayed_locations)}")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n Mission interrupted by user")
        mission.return_to_launch()
    except Exception as e:
        print(f"\n Mission error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            kml_file = export_sprayed_locations_kml(mission.sprayed_locations, "waypoints/sprayed_locations.kml")
            if kml_file:
                print("? KML saved:", kml_file)
                print("? Exported count:", len(mission.sprayed_locations))
        except Exception as e:
            print("? KML export failed (mission continues):", e)


        mission.close_connection()

if __name__ == "__main__":
    main()
