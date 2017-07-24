import numpy as np

import rosbag
import cv2
import array
# from cv_bridge import CvBridge

TOPIC_NAMES = {
    "/ici/ir_camera/image": "thermal",
    "/camera/rgb/image_color": "color",
    "/camera/depth_registered/image_raw": "depth",
    "/correspondance_points": "points"
}
COLOR_FOCAL_LENGTH = 535.0
COLOR_WIDTH = 640
COLOR_HEIGHT = 480
COLOR_PRINCIPAL_X = COLOR_WIDTH/2
COLOR_PRINCIPAL_Y = COLOR_HEIGHT/2
THERMAL_FOCAL_LENGTH = 735.29
THERMAL_WIDTH = 640
THERMAL_HEIGHT = 512
THERMAL_PRINCIPAL_X = THERMAL_WIDTH/2
THERMAL_PRINCIPAL_Y = THERMAL_HEIGHT/2

A = np.array([[THERMAL_FOCAL_LENGTH, 0, THERMAL_PRINCIPAL_X], [0, THERMAL_FOCAL_LENGTH, THERMAL_PRINCIPAL_Y], [0, 0, 1]], np.int32)
# calib_flags = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_PRINCIPAL_POINT  \
        #  + cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6
        #  + cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_ZERO_TANGENT_DIST
calib_flags = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_FIX_FOCAL_LENGTH\
         + cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6
        #  + cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_ZERO_TANGENT_DIST 

bag = rosbag.Bag("calib.bag")

messages = {}
for topic, message, time in bag.read_messages():
    if not time in messages:
        messages[time] = {}
    messages[time][TOPIC_NAMES[topic]] = message

times = sorted(messages.keys())

# bridge = CvBridge()
# time = times[0]
# cv_img = bridge.imgmsg_to_cv2(messages[time]["color"])

objpoints = []
imgpoints = []

for _,frame in messages.items():
    points = eval(frame["points"].data)
    # depths = array.array("H", frame["depth"].data)
    depths = np.fromstring(frame["depth"].data, np.int16).reshape(COLOR_HEIGHT, COLOR_WIDTH)
    frameobjpoints = []
    for i,j in points["color"]:
        depth = depths[int(round(j)), int(round(i))]/1000.0
        x = (i-COLOR_PRINCIPAL_X)*depth/COLOR_FOCAL_LENGTH
        y = (j-COLOR_PRINCIPAL_Y)*depth/COLOR_FOCAL_LENGTH
        z = depth
        frameobjpoints.append([x,y,z])
    objpoints.append(frameobjpoints)
    imgpoints.append(points["thermal"])

# def colortoxyz(i, j):
#     depth = depths[int(round(i)) + int(round(j))*COLOR_WIDTH]/1000.0
#     x = (i-COLOR_PRINCIPAL_X)*depth/COLOR_FOCAL_LENGTH
#     y = (j-COLOR_PRINCIPAL_Y)*depth/COLOR_FOCAL_LENGTH
#     z = depth
#     return x,y,z

MAX_T = 1
ret, mat, dist, rs, ts = cv2.calibrateCamera(np.array(objpoints, np.float32), np.array(imgpoints, np.float32), (COLOR_WIDTH,COLOR_HEIGHT), cameraMatrix=A, flags=calib_flags)

# run second time on filtered points
ts = np.array(ts).sum(axis=2)
ret, mat, dist, rs, ts = cv2.calibrateCamera(np.array(objpoints, np.float32)[(np.abs(ts)<MAX_T).all(axis=1)], np.array(imgpoints, np.float32)[(np.abs(ts)<MAX_T).all(axis=1)], (COLOR_WIDTH,COLOR_HEIGHT), cameraMatrix=A, flags=calib_flags)

dist = dist.flatten()
ts = np.array(ts).sum(axis=2)
rs = np.array(rs).sum(axis=2)
# r = np.array([np.average(rs[:,i,0]) for i in range(3)])
# rstd = np.array([np.std(rs[:,i,0]) for i in range(3)])

# t = np.array([np.average(ts[:,i,0]) for i in range(3)])
# tstd = np.array([np.std(ts[:,i,0]) for i in range(3)])
# ts[(ts<10).all(axis=1).flatten(),:,:]
filtered_ts = ts[(np.abs(ts)<MAX_T).all(axis=1)] # cut out outliers
t = np.mean(filtered_ts, axis=0)
tstd = np.std(filtered_ts, axis=0)

filtered_rs = rs[(np.abs(ts)<MAX_T).all(axis=1)] # cut out outliers from ts
r = np.mean(filtered_rs, axis=0)
rstd = np.std(filtered_ts, axis=0)

R,_ = cv2.Rodrigues(r)
# Rt = np.concatenate((R, np.array([t]).T), axis=1)

constants = np.array((COLOR_WIDTH, COLOR_HEIGHT, COLOR_FOCAL_LENGTH, COLOR_PRINCIPAL_X, COLOR_PRINCIPAL_Y,
             THERMAL_WIDTH, THERMAL_HEIGHT, THERMAL_FOCAL_LENGTH, THERMAL_PRINCIPAL_X, THERMAL_PRINCIPAL_Y, dist, r, t), dtype=object)
# np.save("constants 1 double r", constants)

# def xyztothermal(coords):
#     x,y,z = np.dot(R, coords) + t
#     x = x/z
#     y = y/z
#     r2 = x*x + y*y
#     k = (1 + k1*r2 + k2*r2**2 + k3*r2**3)/(1 + k4*r2 + k5*r2**2 + k6*r2**3)
#     x = x*k + 2*p1*x*y + p2*(r2 + 2*x*x)
#     y = y*k + p1*(r2 + 2*y*y) + 2*p2*x*y
#     i = int(round(THERMAL_FOCAL_LENGTH*x + THERMAL_PRINCIPAL_X))
#     j = int(round(THERMAL_FOCAL_LENGTH*y + THERMAL_PRINCIPAL_Y))
#     return i,j

# def gethands(thermalhands):
#     hands = np.zeros((COLOR_WIDTH, COLOR_HEIGHT), np.bool)
#     for i in range(COLOR_WIDTH):
#         for j in range(COLOR_HEIGHT):
#             xyz = colortoxyz(i,j)
#             u,v = xyztothermal(xyz)
#             hands[i,j] = thermalhands[u,v]
#     return hands

def plot3d(ts):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(ts[:,0], ts[:,1], ts[:,2])
    plt.show()
