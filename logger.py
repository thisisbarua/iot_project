import serial
import csv
import time
import re
from datetime import datetime

# ==========================================
# HARDWARE CONFIGURATION
# ==========================================
PORT = '/dev/ttyACM0'  # <--- NOTE: Change this to /dev/ttyACM1 on Anqur's laptop!
BAUD = 115200
DURATION = 60  # Set to 30 * 60 for the real run

# The Master Dictionary of your hardware
MASTER_MACS = {
    'A': 'AE:F7:83:9A:A9:52:C5:CD',
    'B': '66:3D:FF:91:0C:09:14:A5',
    'C': 'A6:66:F7:7F:8F:C7:AC:A4'
}

def setup_receiver_identity():
    """Prompts the user to identify the current receiver node."""
    print("\n" + "="*45)
    print("   📡 IoT Data Logger - Receiver Setup")
    print("="*45)
    
    while True:
        choice = input("Which Node is THIS laptop connected to? (A, B, or C): ").strip().upper()
        
        # Make it foolproof: handle if you accidentally type "node a" instead of "a"
        if choice.startswith("NODE "):
            choice = choice.replace("NODE ", "").strip()
            
        if choice in MASTER_MACS:
            receiver_node = f"Node_{choice}"
            receiver_mac = MASTER_MACS[choice]
            
            # Automatically create the known senders list (everyone EXCEPT this receiver)
            known_senders = {
                mac: f"Node_{letter}" 
                for letter, mac in MASTER_MACS.items() 
                if letter != choice
            }
            
            print(f"\n✅ Successfully configured as {receiver_node} ({receiver_mac})")
            return receiver_node, receiver_mac, known_senders
        else:
            print("❌ Invalid input. Please type A, B, or C.")

def main():
    # 1. Run the interactive setup prompt
    RECEIVER_NODE, RECEIVER_MAC, KNOWN_SENDERS = setup_receiver_identity()
    
    # 2. Start the data collection
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
        print(f"Listening on {PORT}...")
        print("⏳ Waiting for the first packet from a Sender to create log files...\n")
        
        raw_buffer = []  
        f_csv = None
        f_raw = None
        writer = None
        
        sender_mac = None
        sender_node = None
        start_time = None
        packet_count = 0
        
        last_rssi_map = {}
        curr_rssi = None
        curr_lqi = None
        curr_src_mac = None

        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue
                
            raw_buffer.append(line)
            if f_raw:
                f_raw.write(line + '\n')
                f_raw.flush()

            if "rssi:" in line and "lqi:" in line:
                m_rssi = re.search(r'rssi:\s*(-?\d+)', line)
                m_lqi = re.search(r'lqi:\s*(\d+)', line)
                if m_rssi and m_lqi:
                    curr_rssi = int(m_rssi.group(1))
                    curr_lqi = int(m_lqi.group(1))
                    
            elif "src_l2addr:" in line:
                m_mac = re.search(r'src_l2addr:\s*([A-F0-9:]+)', line, re.IGNORECASE)
                if m_mac:
                    curr_src_mac = m_mac.group(1).upper()
                    
            elif "~~ PKT" in line:
                if curr_rssi is not None and curr_lqi is not None and curr_src_mac is not None:
                    
                    if not sender_mac:
                        sender_mac = curr_src_mac
                        sender_node = KNOWN_SENDERS.get(sender_mac, f"Unknown_{sender_mac[-5:].replace(':','')}")
                        
                        file_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        csv_name = f"data_{sender_node}_to_{RECEIVER_NODE}_{file_ts}.csv"
                        raw_name = f"raw_{sender_node}_to_{RECEIVER_NODE}_{file_ts}.log"
                        
                        print(f"✅ Target Locked: {sender_node} ({sender_mac})")
                        print(f"📁 Creating Logs: {csv_name} & {raw_name}")
                        print("🔴 Recording Started!")
                        
                        f_csv = open(csv_name, 'w', newline='')
                        f_raw = open(raw_name, 'w')
                        
                        for b_line in raw_buffer:
                            f_raw.write(b_line + '\n')
                        f_raw.flush()
                        raw_buffer.clear() 
                        
                        writer = csv.writer(f_csv)
                        writer.writerow(['Sender_Node', 'Sender_MAC', 'Receiver_Node', 'Receiver_MAC', 'Timestamp', 'RSSI', 'LQI', 'Diff_RSSI'])
                        
                        start_time = time.time()
                    
                    if sender_mac == curr_src_mac:
                        prev = last_rssi_map.get(curr_src_mac, None)
                        diff_rssi = (curr_rssi - prev) if prev is not None else 0
                        last_rssi_map[curr_src_mac] = curr_rssi
                        
                        row_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        
                        writer.writerow([sender_node, sender_mac, RECEIVER_NODE, RECEIVER_MAC, row_ts, curr_rssi, curr_lqi, diff_rssi])
                        packet_count += 1
                    
                curr_rssi = None
                curr_lqi = None
                curr_src_mac = None

            if start_time and (time.time() - start_time) >= DURATION:
                break

        print(f"\n✅ Capture complete! {packet_count} packets saved.")
        if f_csv: f_csv.close()
        if f_raw: f_raw.close()
        
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
    finally:
        ser.close()

if __name__ == "__main__":
    main()
