import serial
import csv
import time
import re
from datetime import datetime

# ==========================================
# HARDWARE CONFIGURATION
# ==========================================
PORT = '/dev/ttyACM0'  # <--- Change to /dev/ttyACM1 for Anqur's laptop!
BAUD = 115200

# The Master Dictionary of your hardware
MASTER_MACS = {
    'A': 'AE:F7:83:9A:A9:52:C5:CD',
    'B': '66:3D:FF:91:0C:09:14:A5',
    'C': 'A6:66:F7:7F:8F:C7:AC:A4'
}

def setup_receiver_identity():
    print("\n" + "="*45)
    print("   📡 CALIBRATION MODE - Range Finder")
    print("="*45)
    
    while True:
        choice = input("Which Node is THIS laptop connected to? (A, B, or C): ").strip().upper()
        if choice.startswith("NODE "):
            choice = choice.replace("NODE ", "").strip()
            
        if choice in MASTER_MACS:
            receiver_node = f"Node_{choice}"
            receiver_mac = MASTER_MACS[choice]
            known_senders = {
                mac: f"Node_{letter}" 
                for letter, mac in MASTER_MACS.items() 
                if letter != choice
            }
            print(f"\n✅ Configured as {receiver_node} ({receiver_mac})")
            return receiver_node, receiver_mac, known_senders
        else:
            print("❌ Invalid input. Please type A, B, or C.")

def main():
    RECEIVER_NODE, RECEIVER_MAC, KNOWN_SENDERS = setup_receiver_identity()
    
    try:
        # TIMEOUT = 0.5 is crucial here so the loop doesn't get stuck when signal drops
        ser = serial.Serial(PORT, BAUD, timeout=0.5)
        print(f"Listening on {PORT}...")
        print("⏳ Waiting for the first packet to lock onto a Sender...\n")
        
        raw_buffer = []  
        f_csv = None
        f_raw = None
        writer = None
        
        sender_mac = None
        sender_node = None
        packet_count = 0
        
        # State tracking for the Calibration Feedback
        has_started = False
        in_range = False
        last_packet_time = time.time()
        last_alert_time = time.time()
        
        last_rssi_map = {}
        curr_rssi = None
        curr_lqi = None
        curr_src_mac = None

        while True:
            current_time = time.time()
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            
            # --- OUT OF RANGE LOGIC ---
            # If we have started tracking, and 1.5 seconds have passed without a packet
            if has_started and (current_time - last_packet_time > 1.5):
                if in_range:
                    print(f"\n❌ OUT OF RANGE! Signal lost. Keep walking back...")
                    in_range = False
                    last_alert_time = current_time
                elif (current_time - last_alert_time) >= 2.0:
                    # Print an update every 2 seconds while still out of range
                    print("⚠️ Still out of range... move closer!")
                    last_alert_time = current_time

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
                        
                        csv_name = f"calibration_data_{sender_node}_to_{RECEIVER_NODE}_{file_ts}.csv"
                        raw_name = f"calibration_logger_{sender_node}_to_{RECEIVER_NODE}_{file_ts}.log"
                        
                        print(f"✅ Target Locked: {sender_node} ({sender_mac})")
                        print(f"📁 Calibration Logs: {csv_name} & {raw_name}")
                        print("🚶‍♂️ You can start walking now. Press Ctrl+C to stop.\n")
                        
                        f_csv = open(csv_name, 'w', newline='')
                        f_raw = open(raw_name, 'w')
                        
                        for b_line in raw_buffer:
                            f_raw.write(b_line + '\n')
                        f_raw.flush()
                        raw_buffer.clear() 
                        
                        writer = csv.writer(f_csv)
                        writer.writerow(['Sender_Node', 'Sender_MAC', 'Receiver_Node', 'Receiver_MAC', 'Timestamp', 'RSSI', 'LQI', 'Diff_RSSI'])
                        
                        has_started = True
                        in_range = True
                        last_packet_time = current_time
                    
                    if sender_mac == curr_src_mac:
                        # --- BACK IN RANGE LOGIC ---
                        if not in_range:
                            print(f"\n✅ BACK IN RANGE! (Recovered at RSSI: {curr_rssi}) - Max distance found!")
                            in_range = True
                        
                        # Reset the packet timer
                        last_packet_time = time.time()
                        
                        # Calculate RSSI differences
                        prev = last_rssi_map.get(curr_src_mac, None)
                        diff_rssi = (curr_rssi - prev) if prev is not None else 0
                        last_rssi_map[curr_src_mac] = curr_rssi
                        
                        row_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        
                        writer.writerow([sender_node, sender_mac, RECEIVER_NODE, RECEIVER_MAC, row_ts, curr_rssi, curr_lqi, diff_rssi])
                        packet_count += 1
                        
                        # Optional: Print every 50th packet just to show it's still healthy
                        if packet_count % 50 == 0:
                            print(f"   ... Signal healthy (Current RSSI: {curr_rssi})")
                    
                curr_rssi = None
                curr_lqi = None
                curr_src_mac = None

    except KeyboardInterrupt:
        # --- GRACEFUL EXIT ON CTRL+C ---
        print(f"\n\n🛑 Calibration stopped manually by user.")
        print(f"✅ Capture complete! {packet_count} calibration packets saved.")
        
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
    finally:
        if 'f_csv' in locals() and f_csv: f_csv.close()
        if 'f_raw' in locals() and f_raw: f_raw.close()
        if 'ser' in locals() and ser.is_open: ser.close()

if __name__ == "__main__":
    main()
