import socket
from time import sleep
from busio import I2C
from adafruit_motor import servo
from adafruit_pca9685 import PCA9685
from board import SCL, SDA


def servorotate(servos_list, servos_used, angles_used, s):
    j = 0
    for k in servos_used:
        servos_list[int(k / 3)].angle = angles_used[j]
        j = j + 1
    sleep(s)


i2c = I2C(SCL, SDA)
pca = PCA9685(i2c)
pca.frequency = 50
servos_tot = [0, 3, 6, 9, 12, 15]

for i in servos_tot:
    if i == 15:
        servos_tot[int(i / 3)] = servo.Servo(pca.channels[i])
    else:
        servos_tot[int(i / 3)] = servo.Servo(pca.channels[i], min_pulse=500, max_pulse=2400)

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = ''
port = 12345

server_socket.bind((host, port))
server_socket.listen(1)
print("Server Listening on Port", port)
client_socket, addr = server_socket.accept()
print(f"Connection Established with {addr}")

try:
    while True:
        data = client_socket.recv(1024)

        if not data:
            print("Websocket Communication Closed")
            break

        predicted_class = data.decode("utf-8")
        print("Class: ", predicted_class)

        # rest
        if predicted_class == "0":
            servorotate(
                servos_list=servos_tot,
                servos_used=[3, 6, 9, 12, 15],
                angles_used=[115, 55, 80, 75, 110],
                s=0.9,
            )

        # closed palm
        elif predicted_class == "1":
            servorotate(
                servos_list=servos_tot,
                servos_used=[3, 6, 9, 12, 15],
                angles_used=[0, 180, 0, 180, 25],
                s=0.9,
            )

        # two
        elif predicted_class == "2":
            servorotate(
                servos_list=servos_tot,
                servos_used=[3, 6, 9, 12, 15],
                angles_used=[0, 180, 180, 0, 25],
                s=0.9,
            )

        # open palm
        elif predicted_class == "3":
            servorotate(
                servos_list=servos_tot,
                servos_used=[3, 6, 9, 12, 15],
                angles_used=[180, 0, 180, 0, 180],
                s=0.9,
            )

        # one
        elif predicted_class == "4":
            servorotate(
                servos_list=servos_tot,
                servos_used=[3, 6, 9, 12, 15],
                angles_used=[0, 180, 0, 0, 25],
                s=0.9,
            )

        # three
        elif predicted_class == "5":
            servorotate(
                servos_list=servos_tot,
                servos_used=[3, 6, 9, 12, 15],
                angles_used=[0, 0, 180, 0, 25],
                s=0.9,
            )

        # four
        elif predicted_class == "6":
            servorotate(
                servos_list=servos_tot,
                servos_used=[3, 6, 9, 12, 15],
                angles_used=[180, 0, 180, 0, 25],
                s=0.9,
            )

        # thumbs up
        elif predicted_class == "7":
            servorotate(
                servos_list=servos_tot,
                servos_used=[3, 6, 9, 12, 15],
                angles_used=[0, 180, 0, 180, 180],
                s=0.9,
            )

except KeyboardInterrupt:
    print("Websocket Communication Closed")
    client_socket.close()
    servorotate(
        servos_list=servos_tot,
        servos_used=[3, 6, 9, 12, 15],
        angles_used=[180, 0, 180, 0, 180],
        s=0.9,
    )
    pca.deinit()
    