import serial
import csv
import time
import re

# Settings for the 1-minute pilot
PORT = '/dev/ttyACM0'
BAUD = 115200
DURATION = 60  # 60 seconds
FILE_NAME = 'pilot_1min_data.csv'

def main():
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
        print(f"Starting 1-minute PILOT run... saving to {FILE_NAME}")
        
        with open(FILE_NAME, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'RSSI', 'Diff_RSSI'])
            
            last_rssi = None
            start_time = time.time()
            packet_count = 0
            
            while (time.time() - start_time) < DURATION:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                
                if "rssi:" in line:
                    match = re.search(r'rssi:\s*(-?\d+)', line)
                    if match:
                        current_rssi = int(match.group(1))
                        diff_rssi = (current_rssi - last_rssi) if last_rssi is not None else 0
                        writer.writerow([time.time(), current_rssi, diff_rssi])
                        last_rssi = current_rssi
                        packet_count += 1
            
        print(f"\nPilot complete! Captured {packet_count} packets.")
        
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        ser.close()

if __name__ == "__main__":
    main()
