import time
import math
import serial
import adafruit_bno055

PORT = "/dev/cu.usbserial-FTAEUZSP"
BAUD = 115200


def circular_mean_deg(angles_deg):
    """Media circular de ángulos en grados (para heading)."""
    rads = [math.radians(a) for a in angles_deg]
    s = sum(math.sin(x) for x in rads)
    c = sum(math.cos(x) for x in rads)
    if s == 0 and c == 0:
        return float("nan")
    return (math.degrees(math.atan2(s, c)) + 360.0) % 360.0


def get_orientation(samples=5):
    # 1) Abrir UART del FT232H
    uart = serial.Serial(PORT, baudrate=BAUD, timeout=1)
    sensor = adafruit_bno055.BNO055_UART(uart)

    time.sleep(0.5)  # dar tiempo a despertar

    # 2) Pasar a CONFIG_MODE para poder escribir offsets
    sensor.mode = adafruit_bno055.CONFIG_MODE
    time.sleep(0.05)

    # --- Tus offsets leídos en Arduino ---
    # adafruit_bno055_offsets_t:
    # (ax, ay, az, mx, my, mz, gx, gy, gz, acc_radius, mag_radius)
    accel_offsets = (-33, -37, -20)
    mag_offsets   = (-260, -35, 242)
    gyro_offsets  = (1, -4, 2)
    accel_radius  = 1000
    mag_radius    = 383

    # 3) Cargar offsets en el BNO (la librería ya escribe en 0x55–0x6A)
    sensor.offsets_accelerometer = accel_offsets
    sensor.offsets_magnetometer  = mag_offsets
    sensor.offsets_gyroscope     = gyro_offsets
    sensor.radius_accelerometer  = accel_radius
    sensor.radius_magnetometer   = mag_radius

    # 4) Volver a NDOF (fusión completa)
    sensor.mode = adafruit_bno055.NDOF_MODE
    time.sleep(0.1)

    print("BNO055 listo. Leyendo orientación promedio de 5 muestras...\n")

    hs, rs, ps = [], [], []

    while len(hs) < samples:
        e = sensor.euler  # (heading, roll, pitch) en grados
        if not e or any(v is None for v in e):
            # lectura inválida, intenta otra vez
            continue

        h, r, p = e
        hs.append(h)
        rs.append(r)
        ps.append(p)
        time.sleep(0.05)  # ~20 Hz

    h_avg = circular_mean_deg(hs)

    uart.close()

    return h_avg
