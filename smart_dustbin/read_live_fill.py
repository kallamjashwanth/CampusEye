import serial
import time

def read_current_fill(com_port="COM3", baud_rate=115200):
    try:
        esp = serial.Serial(com_port, baud_rate, timeout=1)
        time.sleep(2)

        start_time = time.time()

        while time.time() - start_time < 5:
            if esp.in_waiting:
                line = esp.readline().decode(errors="ignore").strip()

                if "FILL:" in line:
                    try:
                        fill_value = float(line.split("FILL:")[1])
                        esp.close()
                        return fill_value
                    except:
                        pass

        esp.close()
        return None

    except:
        return None