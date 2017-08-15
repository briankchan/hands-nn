
import array
import numpy as np
import rosbag
import rospy
import cv2
from scipy.misc import imsave
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
import matplotlib.pyplot as plt

OVERLAY_ALPHA = 0.3

THRESHOLD = 3650
# hands1 min: ?
# hands1 medium: 3650 <-
# hands1 max: 3500
# hands2 min: 3750?
# hands2 medium: 3700
# hands2 max: 3650? <-

THERMAL_TOPIC_NAME = "/ici/ir_camera/image"
COLOR_TOPIC_NAME = "/camera/rgb/image_color"
DEPTH_COLOR_NAME = "/camera/depth_registered/image_raw"

# inf = float("inf")
nan = float("nan")

bag = rosbag.Bag("data1.bag")
SKIP = 0#63 * 28
SAVE_NP = True
NP_FILENAME = "hands1-3650"
SAVE_VID = False
FILENAME = "hands1medium.avi"

COLOR_WIDTH, COLOR_HEIGHT, COLOR_FOCAL_LENGTH, COLOR_PRINCIPAL_X, COLOR_PRINCIPAL_Y,\
        THERMAL_WIDTH, THERMAL_HEIGHT, THERMAL_FOCAL_LENGTH, THERMAL_PRINCIPAL_X, THERMAL_PRINCIPAL_Y,\
        (k1, k2, p1, p2, k3, k4, k5, k6), r, t = np.load("constants 1 double r.npy")\

THERMAL_FOCAL_LENGTH = 735.29
# THERMAL_FOCAL_LENGTH = 846.6

# r = np.array([ 0.00036431, -0.01013911, -0.0005117 ])
# t = np.array([ 0.00723754, -0.01400035,  0.01128419])

# first:
# r = np.array([-0.00963569, -0.00013911, -0.0005117 ])
# t = np.array([ 0.00723754, -0.01400035, -0.00871581])

# first 2:
# r = np.array([-0.02963569, -0.00013911, -0.0005117 ])
# t = np.array([ 0.00723754, -0.03400035, -0.00871581])

# second:
# r = np.array([-0.02963569, -0.00013911, -0.0005117 ])
# t = np.array([ 0.00723754, -0.05400035, -0.00871581])

# calib 2:
r = np.array([ 0.06755048,  0.01013327,  0.00100022])
t = np.array([ 0.00139976,  0.03109518, -0.0147689 ])

# calib 3:
# r = np.array([ 0.06391887,  0.04428477, -0.00137299])
# t = np.array([-0.0369808 ,  0.02684998, -0.01524325])


R,_ = cv2.Rodrigues(r)
# flip R and t to account for xyz point ordering being backwards (zyx)
# R_inv = np.flip(np.flip(R, 0), 1).T
R_inv = np.rot90(R, 2).T
t_inv = np.flip(t, 0)

def colortoxyz(i, j, depths):
    depth = depths[int(round(j)), int(round(i))]/1000.0
    x = (i-COLOR_PRINCIPAL_X)*depth/COLOR_FOCAL_LENGTH
    y = (j-COLOR_PRINCIPAL_Y)*depth/COLOR_FOCAL_LENGTH
    z = depth
    return x,y,z

def xyztothermal(coords):
    x,y,z = np.dot(R, coords) + t
    # x,y,z = coords + t
    x = x/z
    y = y/z
    # r2 = x*x + y*y
    # k = (1 + k1*r2 + k2*r2**2 + k3*r2**3)/(1 + k4*r2 + k5*r2**2 + k6*r2**3)
    # x = x*k + 2*p1*x*y + p2*(r2 + 2*x*x)
    # y = y*k + p1*(r2 + 2*y*y) + 2*p2*x*y
    i = int(round(THERMAL_FOCAL_LENGTH*x + THERMAL_PRINCIPAL_X))
    j = int(round(THERMAL_FOCAL_LENGTH*y + THERMAL_PRINCIPAL_Y))
    return i,j

def gethandsold(thermalhands, depths):
    hands = np.zeros((COLOR_HEIGHT, COLOR_WIDTH), thermalhands.dtype)

    for j in range(COLOR_HEIGHT):
        for i in range(COLOR_WIDTH):
            xyz = colortoxyz(i,j,depths)
            if xyz[2] < 1e-10:
                continue
            u,v = xyztothermal(xyz)
            if u<0 or v<0 or u>=THERMAL_WIDTH or v>=THERMAL_HEIGHT:
                continue
            # try:
            hands[j,i] = thermalhands[v,u] # width vs height what on earth
            # except IndexError:
            #     pass
    return hands

# generate matrix of coordinates offset by principal
coords = np.array(np.meshgrid(np.arange(-COLOR_PRINCIPAL_Y, COLOR_PRINCIPAL_Y), np.arange(-COLOR_PRINCIPAL_X, COLOR_PRINCIPAL_X), indexing="ij")).transpose([1,2,0])

def gethands(thermalhands, depths):
    # convert uv to xyz of color camera frame
    zs = depths.reshape(COLOR_HEIGHT, COLOR_WIDTH, 1)/1000.
    zs[zs<1e-10] = nan
    yxs = coords * zs / COLOR_FOCAL_LENGTH
    zyxs = np.concatenate((zs, yxs), axis=2)
    # transform to thermal camera frame
    thermal_zyxs = zyxs.dot(R_inv) + t_inv
    # convert xyz back to uv
    thermal_vus = thermal_zyxs[:,:,[1,2]] / thermal_zyxs[:,:,[0]]
    thermal_vus_offset = np.round(thermal_vus * THERMAL_FOCAL_LENGTH + [THERMAL_PRINCIPAL_Y + 1, THERMAL_PRINCIPAL_X + 1])
    thermal_vus_offset[np.isnan(thermal_vus_offset)] = 0 # depth = 0 -> default value (0 or false)
    thermal_vus_offset = thermal_vus_offset.astype(np.int)

    padded = np.pad(thermalhands, 1, "constant")
    indices = thermal_vus_offset.reshape(-1,2).T
    return padded.take(np.ravel_multi_index(indices, padded.shape, mode="clip")).reshape(COLOR_HEIGHT, COLOR_WIDTH)

    # hands = thermalhands[thermal_vus]

# subscribers = [message_filters.Subscriber(TOPICS[0][0], TOPICS[0][1])]

# ts = message_filters.ApproximateTimeSynchronizer(subscribers, 10, 1./15)
# messages = []
# def callback(color, thermal, depth):
#     messages.append({
#         "color": color,
#         "thermal": thermal,
#         "depth": depth
#         })
# ts.registerCallback(callback)
# rospy.spin()

# topicnames,_ = zip(*TOPICS)
count = 0
prevdepth = None
prevthermal = None
# prevcolor = None
output = []

messages = bag.read_messages()

def getnextframe(messages, prevdepth, prevthermal):
    topic, message, time = messages.next()
    while topic != COLOR_TOPIC_NAME:
        if topic == DEPTH_COLOR_NAME:
            prevdepth = np.fromstring(message.data, np.int16).reshape(COLOR_HEIGHT, COLOR_WIDTH)
        elif topic == THERMAL_TOPIC_NAME:
            prevthermal = np.fromstring(message.data, np.int16).reshape(THERMAL_HEIGHT, THERMAL_WIDTH)
        topic, message, time = messages.next()
    # if prevdepth is None or prevthermal is None:
    #     return np.zeros((COLOR_HEIGHT, COLOR_WIDTH), np.bool) # zero hands
    return np.fromstring(message.data, np.uint8).reshape(COLOR_HEIGHT, COLOR_WIDTH, 3), prevdepth, prevthermal

def plottimes(messages):
    thermal_times = []
    color_times = []
    depth_times = []
    for topic, msg, time in messages:
        if topic == THERMAL_TOPIC_NAME:
            lst = thermal_times
        elif topic == COLOR_TOPIC_NAME:
            lst = color_times
        elif topic == DEPTH_COLOR_NAME:
            lst = depth_times
        else:
            continue
        lst.append(time.to_time())
    plt.plot(thermal_times, np.zeros_like(thermal_times), ".")
    plt.plot(color_times, np.ones_like(color_times), ".")
    plt.plot(depth_times, np.zeros_like(depth_times) + 2, ".")
    plt.show()

# out = getnexthands(messages)
# print(len(out))
color, prevdepth, prevthermal = getnextframe(messages, prevdepth, prevthermal)
while prevdepth is None or prevthermal is None:
    color, prevdepth, prevthermal = getnextframe(messages, prevdepth, prevthermal)

for _ in range(SKIP):
    color, prevdepth, prevthermal = getnextframe(messages, prevdepth, prevthermal)

labels = gethands(prevthermal > THRESHOLD, prevdepth)

overlaycolor = np.array([255, 0, 0], dtype=np.uint8)
def overlay(labels, image):
    # overlay = np.concatenate((labels.astype(np.uint8).reshape(COLOR_HEIGHT, COLOR_WIDTH, 1) * 255,
    #                           np.zeros((COLOR_HEIGHT, COLOR_WIDTH, 2), dtype=np.uint8)),
    #                          axis=2)
    # return cv2.addWeighted(overlay, OVERLAY_ALPHA, image, 1-OVERLAY_ALPHA, 0)
    image[labels] = overlaycolor
    return image


def savevideo_old(filename, messages, prevdepth, prevthermal, labels, color):
    count = 1
    writer = cv2.VideoWriter(filename, cv2.cv.CV_FOURCC(*'XVID'), 28, (640, 480))
    overlayed = overlay(labels, color)
    writer.write(overlayed)
    while True:
        count += 1
        if count % 50 == 0:
            print(count)
        # if count == 300:
        #     break
        try:
            color, prevdepth, prevthermal = getnextframe(messages, prevdepth, prevthermal)
        except StopIteration:
            break
        labels = gethands(prevthermal > THRESHOLD, prevdepth)
        overlayed = overlay(labels, color)
        writer.write(overlayed)

def label_overlay_generator(messages, prevdepth, prevthermal, labels, color):
    yield overlay(labels, color)

    while True:
        try:
            color, prevdepth, prevthermal = getnextframe(messages, prevdepth, prevthermal)
            labels = gethands(prevthermal > THRESHOLD, prevdepth)
            yield overlay(labels, color)
        except StopIteration:
            return


def makegrid(color, depth, thermal, labels):
    depth = (depth / (4000/255)).astype(np.uint8) # rescale
    depth = np.expand_dims(depth, -1).repeat(3, axis=-1)

    # thermal is 512x640; everything else is 480x640
    thermal = thermal[16:-16]
    thermal = ((thermal - 2875) / (1126./255)).astype(np.uint8) # rescale
    thermal = np.expand_dims(thermal, -1).repeat(3, axis=-1)

    labels = np.expand_dims(labels, -1).repeat(3, axis=-1)
    labels = (labels * 255).astype(np.uint8)

    top = np.concatenate((color, depth), axis=1)
    bot = np.concatenate((thermal, labels), axis=1)
    return np.concatenate((top, bot), axis=0)

def grid_generator(messages, color, prevdepth, prevthermal, labels):
    yield makegrid(color, prevdepth, prevthermal, labels)
    i = 1
    while True:
        try:
            color, prevdepth, prevthermal = getnextframe(messages, prevdepth, prevthermal)
            labels = gethands(prevthermal > THRESHOLD, prevdepth)
            yield makegrid(color, prevdepth, prevthermal, labels)
            i += 1
        except StopIteration:
            return

def savevideo(filename, generator, shape):
    writer = cv2.VideoWriter(filename, cv2.cv.CV_FOURCC(*'XVID'), 28, shape)
    i = 0
    for frame in generator:
        if i % 50 == 0:
            print(i)
        i += 1
        writer.write(frame)

def savenp(filename, messages, prevdepth, prevthermal, labels, color):
    count = 0
    labels = [labels]
    images = [color]
    while True:
        count += 1
        if count % 50 == 0:
            print(count)
        try:
            color, prevdepth, prevthermal = getnextframe(messages, prevdepth, prevthermal)
        except StopIteration:
            break
        labels.append(gethands(prevthermal > THRESHOLD, prevdepth))
        images.append(color)
    # np.savez(filename, images=images, labels=labels)
    np.save(filename + "-images", images)
    np.save(filename + "-labels", labels)

step = 0.01
def onkeypress(event):
    global step
    global r
    global t
    global R
    global R_inv
    global t_inv
    global labels
    global color, prevdepth, prevthermal

    if event.key == "q":
        r[0] += step
    elif event.key == "w":
        r[1] += step
    elif event.key == "e":
        r[2] += step
    elif event.key == "a":
        r[0] -= step
    elif event.key == "s":
        r[1] -= step
    elif event.key == "d":
        r[2] -= step
    elif event.key == "u":
        t[0] += step
    elif event.key == "i":
        t[1] += step
    elif event.key == "o":
        t[2] += step
    elif event.key == "j":
        t[0] -= step
    elif event.key == "k":
        t[1] -= step
    elif event.key == "l":
        t[2] -= step
    elif event.key == " ":
        color, prevdepth, prevthermal = getnextframe(messages, prevdepth, prevthermal)
    elif event.key == "[":
        step /= 10
    elif event.key == "]":
        step *= 10
    print(r, t, step)
    R, _ = cv2.Rodrigues(r)
    R_inv = np.rot90(R, 2).T
    t_inv = np.flip(t, 0)

    labels = gethands(prevthermal > THRESHOLD, prevdepth)
    overlayed = overlay(labels, color)

    event.canvas.figure.gca().imshow(overlayed)
    event.canvas.draw()
    # TODO: clear canvas before redrawing


def tryrts():
    plt.rcParams["keymap.all_axes"] = ""
    plt.rcParams["keymap.save"] = "ctrl+s"
    plt.rcParams["keymap.zoom"] = ""
    plt.rcParams["keymap.xscale"] = ""
    plt.rcParams["keymap.yscale"] = ""

    fig = plt.figure()
    fig.canvas.mpl_connect("key_press_event", onkeypress)
    overlayed = overlay(labels, color)
    plt.imshow(overlayed)
    plt.show()

if __name__ == "__main__":
    if SAVE_NP:
        savenp(NP_FILENAME, messages, prevdepth, prevthermal, labels, color)
    elif SAVE_VID:
        savevideo(FILENAME, messages, prevdepth, prevthermal, labels, color)
    else:
        tryrts()

def imshow(img):
    plt.imshow(img)
    plt.show()
    plt.clf()

# for topic, message, time in bag.read_messages():

#     if topic == DEPTH_COLOR_NAME:
#         prevdepth = np.fromstring(message.data, np.int16).reshape(COLOR_HEIGHT, COLOR_WIDTH)
#     elif topic == THERMAL_TOPIC_NAME:
#         prevthermal = np.fromstring(message.data, np.int16).reshape(THERMAL_HEIGHT, THERMAL_WIDTH)
#     elif topic == COLOR_TOPIC_NAME:
#         if prevdepth is None or prevthermal is None:
#             output.append(np.zeros((COLOR_HEIGHT, COLOR_WIDTH), np.bool))
#             continue

#         img = np.fromstring(message.data, np.uint8).reshape(COLOR_HEIGHT, COLOR_WIDTH, 3)

#         count += 1
#         print(count)
#         transformed = gethands(prevthermal > THRESHOLD, prevdepth)
#         # transformed = gethands(prevthermal, prevdepth)
#         if count < 3:
#             # imsave("image.jpg", transformed)
#             overlay = np.concatenate((transformed.astype(np.uint8).reshape(COLOR_HEIGHT, COLOR_WIDTH, 1) * 255, np.zeros((COLOR_HEIGHT, COLOR_WIDTH, 2), dtype=np.uint8)), axis=2)
#             plt.imshow(cv2.addWeighted(overlay, OVERLAY_ALPHA, img, 1-OVERLAY_ALPHA, 0))
#             plt.show()
#         output.append(transformed)
#         if count >= 3: break

# np.save("data", output)
    # try:
    #     i = topicnames.index(topic)
    #     # print(i)
    #     if count <= 500:
    #         subscribers[i].signalMessage(message)
    # except:
    #     pass
# if TOPIC_NAMES[topic] == "thermal":
# if TOPIC_NAMES[topic] info== "thermalinfo":
#     messages[time] = message[]

# times = sorted(messages.keys())

# time = times[0]

# thermal = np.fromstring(messages[time]["thermal"].data, np.int16).reshape(512, 640)
# imsave("image.jpg", thermal)
# filtered = thermal > THRESHOLD
