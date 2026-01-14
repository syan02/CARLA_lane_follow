import carla
import numpy as np
import cv2
import time

# Connect to CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Get blueprint of vehicle
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]

# Spawn vehicle at random spawn point
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# Add camera sensor
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '640')
camera_bp.set_attribute('image_size_y', '480')
camera_bp.set_attribute('fov', '110')
camera_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# Lane detection callback
def process_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    cv2.imshow("Lane Detection", edges)
    cv2.waitKey(1)

camera.listen(lambda image: process_image(image))

# Simple automatic control (just forward)
vehicle.apply_control(carla.VehicleControl(throttle=0.5))

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    camera.stop()
    vehicle.destroy()
    cv2.destroyAllWindows()
