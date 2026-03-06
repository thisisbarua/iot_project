import serial
import csv
import time
import re

# Configuration for Laptop 2 (S3 / Node C)
PORT = '/dev/ttyACM1'  # Updated to your verified port
BAUD = 115200
DURATION = 60  # Set to 60 for the quick 1-minute pilot run
CSV_FILE = 'parsed_data_NodeC.csv'  # Unique filename for Receiver C
RAW_FILE = 'raw_dump_NodeC.log'     # Unique filename for Receiver C

def main():
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
        print(f"Listening on {PORT}...")
        print(f"Saving RAW data to -> {RAW_FILE}")
        print(f"Saving CSV data to -> {CSV_FILE}")
        
        with open(CSV_FILE, 'w', newline='') as f_csv, open(RAW_FILE, 'w') as f_raw:
            writer = csv.writer(f_csv)
            # Schema matches your project image requirements, plus ML necessities
            writer.writerow(['Node_ID', 'MAC_Address', 'Timestamp', 'RSSI', 'LQI', 'Diff_RSSI'])
            
            # State tracking variables
            last_rssi_map = {}
            curr_rssi = None
            curr_lqi = None
            curr_mac = None
            
            start_time = time.time()
            packet_count = 0
            
            while (time.time() - start_time) < DURATION:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                
                if not line:
                    continue
                    
                # 1. THE RAW LOGGER: Write every single line to the backup file
                f_raw.write(line + '\n')
                f_raw.flush()  # Forces write to disk immediately in case of a crash
                
                # 2. Extract RSSI and LQI
                if "rssi:" in line and "lqi:" in line:
                    m_rssi = re.search(r'rssi:\s*(-?\d+)', line)
                    m_lqi = re.search(r'lqi:\s*(\d+)', line)
                    if m_rssi and m_lqi:
                        curr_rssi = int(m_rssi.group(1))
                        curr_lqi = int(m_lqi.group(1))
                        
                # 3. Extract MAC Address
                elif "src_l2addr:" in line:
                    m_mac = re.search(r'src_l2addr:\s*([A-F0-9:]+)', line)
                    if m_mac:
                        curr_mac = m_mac.group(1)
                        
                # 4. Trigger the CSV Write on the end-of-packet marker
                elif "~~ PKT" in line:
                    if curr_rssi is not None and curr_lqi is not None and curr_mac is not None:
                        # Create a readable Node ID (e.g., "Node_C5CD") from the MAC
                        node_id = f"Node_{curr_mac[-5:].replace(':', '')}"
                        
                        # Calculate Differentiated RSSI independently for each MAC address
                        prev = last_rssi_map.get(curr_mac, None)
                        diff_rssi = (curr_rssi - prev) if prev is not None else 0
                        last_rssi_map[curr_mac] = curr_rssi
                        
                        # Write the row matching your image layout
                        writer.writerow([node_id, curr_mac, time.time(), curr_rssi, curr_lqi, diff_rssi])
                        packet_count += 1
                        
                    # Reset variables for the next incoming packet
                    curr_rssi = None
                    curr_lqi = None
                    curr_mac = None

        print(f"\nCapture complete! {packet_count} packets processed successfully.")
        
    except Exception as e:
        print(f"\nFatal Error: {e}")
    finally:
        ser.close()

if __name__ == "__main__":
    main()
