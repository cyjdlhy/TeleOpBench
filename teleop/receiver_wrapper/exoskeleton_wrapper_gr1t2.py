import numpy as np
import time
import serial
import threading
from threading import Event, Thread, Lock
from dynamixel_sdk import *
import lcm
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from lcm_with_hand.xsens_gr1_lcmt import xsens_gr1_lcmt

# Dynamixel Configuration
ADDR_PRO_TORQUE_ENABLE = 64
ADDR_PRO_GOAL_POSITION = 116
ADDR_PRO_PRESENT_POSITION = 132
LEN_PRO_GOAL_POSITION = 4
LEN_PRO_PRESENT_POSITION = 4
PROTOCOL_VERSION = 2.0
TORQUE_ENABLE = True
TORQUE_DISABLE = False
MINIMUM_POSITION_VALUE = 20
MAXIMUM_POSITION_VALUE = 4080

# Basic Configuration
INIT_ID = list(np.arange(1, 15))
BAUDRATE = 57600
DEVICENAME = '/dev/ttyUSB0'

# Gloves Configuration
BAUD_RATE = 9600
RIGHT_HAND_SERIAL_PORT = '/dev/ttyUSB1'
LEFT_HAND_SERIAL_PORT = '/dev/ttyUSB2'
TIMEOUT = 1
START_BYTE = 0x01
END_BYTE = 0x02

# Convert unit constants
ENCODE = 1
ANGLE = 2
RADIAN = 3

class ExoskeletonWrapper:
    def __init__(self, ID=INIT_ID, Baudrate=BAUDRATE, Device=DEVICENAME, 
                 Torque=TORQUE_DISABLE, Unit=ANGLE):
        """
        Initialize ExoskeletonWrapper for GR1T2
        
        Args:
            ID: List of Dynamixel servo IDs
            Baudrate: Communication baudrate
            Device: Serial device path
            Torque: Initial torque state
            Unit: Unit for angle conversion (ANGLE/RADIAN/ENCODE)
        """
        self.ID = ID if ID is not None else []
        self.Baudrate = Baudrate
        self.Device = Device
        self.Torque = Torque
        self.Unit = Unit
        
        # Data storage
        self.Servo_value = np.zeros(len(self.ID), dtype=int)
        self.send_value = np.zeros(41, dtype=np.double)
        self.action = np.zeros(41, dtype=np.double)
        
        # Threading control
        self.lock = Lock()
        self.thread_stop_event = Event()
        
        # Initialize Dynamixel communication
        self._init_dynamixel()
        
        # Initialize gloves communication
        self._init_gloves()
        
        # Initialize LCM
        self.lc = lcm.LCM("udpm://239.255.255.255:7667?ttl=1")
        
        # Hand calibration data
        self._init_hand_calibration()
        
        print("Initialization completed!")

    def _init_dynamixel(self):
        """Initialize Dynamixel servo communication"""
        self.portHandler = PortHandler(self.Device)
        self.packetHandler = PacketHandler(PROTOCOL_VERSION)
        self.groupSyncWrite = GroupSyncWrite(self.portHandler, self.packetHandler, 
                                           ADDR_PRO_GOAL_POSITION, LEN_PRO_GOAL_POSITION)
        self.groupSyncRead = GroupSyncRead(self.portHandler, self.packetHandler, 
                                         ADDR_PRO_PRESENT_POSITION, LEN_PRO_PRESENT_POSITION)
        
        # Open port and set baudrate
        if not self.portHandler.openPort():
            raise RuntimeError("Error: Unable to open the port.")
        print("Port opened successfully.")
        
        if not self.portHandler.setBaudRate(self.Baudrate):
            raise RuntimeError("Error: Unable to set the baudrate.")
        print("Baudrate set successfully.")
        
        self.set_torque(self.Torque)

    def _init_gloves(self):
        """Initialize gloves serial communication"""
        self.Right_glove_ser = serial.Serial(RIGHT_HAND_SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
        self.Left_glove_ser = serial.Serial(LEFT_HAND_SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
        
        # Initialize hand data storage
        self.Right_hand_data = np.zeros(15)
        self.Left_hand_data = np.zeros(15)

    def _init_hand_calibration(self):
        """Initialize hand calibration parameters for GR1T2"""
        self.Left_hand_min_data = [140, 135, 135, 135, 125, 135, 128, 130, 133, 135, 128]
        self.Left_hand_max_data = [162, 195, 160, 195, 161, 195, 158, 195, 164, 195, 158]
        self.Right_hand_min_data = [134, 138, 140, 138, 128, 130, 135, 130, 136, 133, 140]
        self.Right_hand_max_data = [165, 194, 162, 195, 175, 190, 191, 195, 186, 195, 180]

        self.GR1_Left_hand_min_data = [-1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.GR1_Left_hand_max_data = [0, 1.22, 1.23, -1.57, -1.74, -1.57, -1.74, 
                                       -1.57, -1.74, -1.57, -1.74]
        self.GR1_Right_hand_min_data = [-1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.GR1_Right_hand_max_data = [0.0, 1.22, 1.23, -1.57, -1.74, -1.57, -1.74, 
                                        -1.57, -1.74, -1.57, -1.74]

    def broadcast_ping(self):
        """Broadcast ping to detect available Dynamixel servos"""
        servo_data_list, servo_comm_result = self.packetHandler.broadcastPing(self.portHandler)
        if servo_comm_result != COMM_SUCCESS:
            raise RuntimeError(f"Communication error: {self.packetHandler.getTxRxResult(servo_comm_result)}")

        print("Broadcast Dynamixel :")
        for servo_id in servo_data_list:
            print(f"Dynamixel ID {servo_id}.")

    def sync_read_positions(self):
        """Continuously read servo positions and process data"""
        # Add all Dynamixel IDs to the GroupSyncRead
        for servo_id in self.ID:
            servo_addparam_result = self.groupSyncRead.addParam(servo_id)
            if not servo_addparam_result:
                raise RuntimeError(f"Dynamixel ID {servo_id} groupSyncRead addparam failed.")

        if self.Torque:
            self.set_torque(torque_state=TORQUE_DISABLE)
            self.Torque = TORQUE_DISABLE

        while not self.thread_stop_event.is_set():
            try:
                # Read servo positions
                servo_comm_result = self.groupSyncRead.txRxPacket()
                if servo_comm_result != COMM_SUCCESS:
                    continue
                
                # Update servo values
                for servo_id in self.ID:
                    if self.groupSyncRead.isAvailable(servo_id, ADDR_PRO_PRESENT_POSITION, 
                                                    LEN_PRO_PRESENT_POSITION):
                        present_position = self.groupSyncRead.getData(servo_id, 
                                                                    ADDR_PRO_PRESENT_POSITION, 
                                                                    LEN_PRO_PRESENT_POSITION)
                        modify_position = self.convert_result(present_position, self.Unit)
                        
                        with self.lock:
                            self.Servo_value[servo_id - 1] = modify_position

                # Process and map data
                self._process_data()
                
            except Exception as e:
                print(f"Error in sync_read_positions: {e}")
                continue

            time.sleep(0.001)

    def _process_data(self):
        """Process and map all sensor data"""
        # Initialize send_value
        self.send_value[0:3] = [0, 0, 0]
        self.send_value[3] = 0.74

        # Process hand data
        self._process_hand_data()
        
        # Process exoskeleton data
        self._process_exoskeleton_data()
        
        # Map to final action
        self._map_to_action()

    def _process_hand_data(self):
        """Process right and left hand glove data"""
        # Right hand processing
        index = [0, 1, 2, 4, 5, 7, 8, 10, 11, 13, 14]
        Right_hand_val = self.Right_hand_data[index]
        
        Right_hand_limit_val = [
            max(self.Right_hand_min_data[i], min(self.Right_hand_max_data[i], Right_hand_val[i]))
            for i in range(len(Right_hand_val))
        ]
        Right_hand_map_val = [
            self.map_to_range(Right_hand_limit_val[i], self.Right_hand_min_data[i], 
                            self.Right_hand_max_data[i], self.GR1_Right_hand_min_data[i], 
                            self.GR1_Right_hand_max_data[i])
            for i in range(len(Right_hand_limit_val))
        ]
        self.send_value[-11:] = Right_hand_map_val[:]

        # Left hand processing
        Left_hand_val = self.Left_hand_data[index]
        Left_hand_limit_val = [
            max(self.Left_hand_min_data[i], min(self.Left_hand_max_data[i], Left_hand_val[i]))
            for i in range(len(Left_hand_val))
        ]
        Left_hand_map_val = [
            self.map_to_range(Left_hand_limit_val[i], self.Left_hand_min_data[i], 
                            self.Left_hand_max_data[i], self.GR1_Left_hand_min_data[i], 
                            self.GR1_Left_hand_max_data[i])
            for i in range(len(Left_hand_limit_val))
        ]
        self.send_value[11:22] = Left_hand_map_val[:]

    def _process_exoskeleton_data(self):
        """Process exoskeleton servo data"""
        # Apply transformations to servo values
        self.Servo_value -= 180
        self.Servo_value[1] -= 30
        self.Servo_value[8] += 30
        self.Servo_value[[2, 3, 4, 6, 7, 9, 11, 12, 13]] = -self.Servo_value[[2, 3, 4, 6, 7, 9, 11, 12, 13]]
        self.Servo_value = self.Servo_value * np.pi / 180.0
        self.Servo_value = self.Servo_value[[0, 1, 2, 3, 4, 6, 5, 7, 8, 9, 10, 11, 13, 12]]
        
        # Map to send_value
        self.send_value[4:11] = self.Servo_value[:7]   # Left arm
        self.send_value[22:29] = self.Servo_value[7:]  # Right arm

    def _map_to_action(self):
        """Map processed data to final action array for GR1T2"""
        # Left upper body
        self.action[0:7] = self.send_value[4:11]
        
        # Left hand - reshuffle from .urdf preview sequence to real (isaacsim) sequence
        self.action[7:18] = self.send_value[[14, 15, 16, 17, 20, 21, 18, 19, 11, 12, 13]]
        
        # Right upper body
        self.action[18:25] = self.send_value[22:29]
        
        # Right hand
        self.action[25:36] = self.send_value[[33, 34, 35, 36, 39, 40, 37, 38, 30, 31, 32]]

    def read_Right_glove_data(self):
        """Read right glove serial data"""
        while not self.thread_stop_event.is_set():
            try:
                byte_data = self.Right_glove_ser.read(1)
                if byte_data and byte_data[0] == START_BYTE:
                    data = []
                    for _ in range(15):
                        data_byte = self.Right_glove_ser.read(1)
                        if data_byte:
                            data.append(data_byte[0])
                    
                    end_byte = self.Right_glove_ser.read(1)
                    if end_byte and end_byte[0] == END_BYTE:
                        self.Right_hand_data = np.array(data)
                    else:
                        print("RIGHT_END_BYTE ERROR")
            except Exception as e:
                print(f"Error reading right glove: {e}")
                continue

    def read_Left_glove_data(self):
        """Read left glove serial data"""
        while not self.thread_stop_event.is_set():
            try:
                byte_data = self.Left_glove_ser.read(1)
                if byte_data and byte_data[0] == START_BYTE:
                    data = []
                    for _ in range(15):
                        data_byte = self.Left_glove_ser.read(1)
                        if data_byte:
                            data.append(data_byte[0])
                    
                    end_byte = self.Left_glove_ser.read(1)
                    if end_byte and end_byte[0] == END_BYTE:
                        self.Left_hand_data = np.array(data)
                    else:
                        print("LEFT_END_BYTE ERROR")
            except Exception as e:
                print(f"Error reading left glove: {e}")
                continue

    def start_receiving(self):
        """Start all data receiving threads"""
        self.read_thread = Thread(target=self.sync_read_positions)
        self.read_thread.daemon = True
        self.read_thread.start()
        print("'sync_read_positions' thread started")

        self.right_glove_thread = Thread(target=self.read_Right_glove_data)
        self.right_glove_thread.daemon = True
        self.right_glove_thread.start()
        print("'read_Right_glove_data' thread started")

        self.left_glove_thread = Thread(target=self.read_Left_glove_data)
        self.left_glove_thread.daemon = True
        self.left_glove_thread.start()
        print("'read_Left_glove_data' thread started")

    def stop_receiving(self):
        """Stop all receiving threads"""
        self.thread_stop_event.set()
        if hasattr(self, 'read_thread'):
            self.read_thread.join()
        if hasattr(self, 'right_glove_thread'):
            self.right_glove_thread.join()
        if hasattr(self, 'left_glove_thread'):
            self.left_glove_thread.join()
        print("All threads stopped.")

    def set_torque(self, torque_state=TORQUE_ENABLE):
        """Set torque state for all servos"""
        with self.lock:
            for servo_id in self.ID:
                servo_comm_result, servo_error = self.packetHandler.write1ByteTxRx(
                    self.portHandler, servo_id, ADDR_PRO_TORQUE_ENABLE, torque_state
                )
                if servo_comm_result != COMM_SUCCESS:
                    raise RuntimeError(f"Communication error: {self.packetHandler.getTxRxResult(servo_comm_result)}")
                elif servo_error != 0:
                    raise RuntimeError(f"Dynamixel ID {servo_id}: {self.packetHandler.getRxPacketError(servo_error)}")
            
            self.Torque = TORQUE_ENABLE if torque_state else TORQUE_DISABLE

    def validate_position(self, position):
        """Validate servo position is within allowed range"""
        min_pos = MINIMUM_POSITION_VALUE
        max_pos = MAXIMUM_POSITION_VALUE
        if not (min_pos <= position <= max_pos):
            raise ValueError(f"Position Error: {position} out the range of ({min_pos}-{max_pos}).")

    def convert_result(self, position, Unit, bits=32):
        """Convert raw position data to specified unit"""
        if position >= 2**(bits - 1):
            position -= 2**bits
        if Unit == ANGLE:
            return position / 4096.0 * 360
        elif Unit == RADIAN:
            return position / 2048.0 * np.pi
        else:
            return position

    def get_torque_value(self):
        """Get current torque state"""
        return self.Torque

    def get_servo_value(self, servo_id):
        """Get current servo value for specific ID"""
        with self.lock:
            return self.Servo_value[servo_id - 1]

    def map_to_range(self, value, old_min, old_max, new_min, new_max):
        """Map value from old range to new range"""
        return (value - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

    def publish(self):
        """Publish action data via LCM"""
        upper_action = xsens_gr1_lcmt()
        upper_action.action = self.action.astype(np.double)
        self.lc.publish("upper_action", upper_action.encode())

    def get_data(self):
        """Get current processed data"""
        return self.send_value

    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.stop_receiving()
        except:
            pass

        try:
            self.set_torque(torque_state=TORQUE_DISABLE)
        except:
            pass

        if hasattr(self, 'portHandler') and self.portHandler.isOpen():
            self.portHandler.closePort()
            print("Port closed.")

if __name__ == "__main__":
    exoskeleton = ExoskeletonWrapper()
    
    try:
        exoskeleton.broadcast_ping()
        exoskeleton.start_receiving()
        while True:
            exoskeleton.get_q()
            exoskeleton.publish()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        exoskeleton.stop_receiving()
        del exoskeleton

