import math
import numpy as np
import cv2
from misc import chunks

DATA1 = "data/hands1-3650"
DATA2 = "data/hands2-3650"
IMAGES = DATA1 + "-images-500.npy"
LABELS = DATA1 + "-labels-500.npy"

IMAGES_FULL = DATA1 + "-images.npy"
LABELS_FULL = DATA1 + "-labels.npy"

IMAGES2_FULL = DATA2 + "-images.npy"
LABELS2_FULL = DATA2 + "-labels.npy"


IMG = np.load(IMAGES)
LAB = np.load(LABELS)
TRAIN = range(400)
TEST = range(400, len(IMG))
# TRAIN_IMG = IMG[:400]
# TRAIN_LAB = LAB[:400]
# TEST_IMG = IMG[400:]
# TEST_LAB = LAB[400:]

def split(ranges, test_chunks, num_chunks):
    test_indices = []
    train_indices = []
    for indices in ranges:
        chunk_size = math.ceil(len(indices) / num_chunks)
        chunks = np.array(list(chunks(indices, chunk_size)))
        test_indices += chunks[test_chunks].tolist()
        train_indices += np.delete(chunks, test_chunks, axis=0).tolist()

    test_indices = np.concatenate(test_indices)
    train_indices = np.concatenate(train_indices)

    return train_indices, test_indices

def get_dataset(dataset, use_dev=True):
    if dataset == 0:
        return IMG, LAB, TRAIN, TEST
    elif dataset == 1:
        images = np.load(IMAGES_FULL)[:-130]
        labels = np.load(LABELS_FULL)[:-130]

        count = len(images)
        half = count // 2
        six_tenths = count // 10 * 6
        nine_tenths = count // 10 * 9

        train = [range(half), range(six_tenths, nine_tenths)]
        test = range(half, six_tenths) if use_dev else range(nine_tenths, count)
        train = np.concatenate(train)
        return images, labels, train, test
    elif dataset == 2:
        images = np.load(IMAGES2_FULL)
        labels = np.load(LABELS2_FULL)

        count = len(images)
        half = count // 2
        six_tenths = count // 10 * 6
        nine_tenths = count // 10 * 9

        train = [range(half), range(six_tenths, nine_tenths)]
        test = range(half, six_tenths) if use_dev else range(nine_tenths, count)
        train = np.concatenate(train)
        return images, labels, train, test
    elif dataset == 3:
        images1 = np.load(IMAGES_FULL)[:-130]
        labels1 = np.load(LABELS_FULL)[:-130]
        images2 = np.load(IMAGES2_FULL)
        labels2 = np.load(LABELS2_FULL)

        images = np.concatenate([images1, images2])
        labels = np.concatenate([labels1, labels2])

        count1 = len(images1)
        split1 = count1 // 10 * 9
        count2 = len(images2)
        split2 = count2 // 10 * 9

        train = [range(split1), range(count1, count1+split2)]
        test = [range(split1, count1), range(count1+split2, count1+count2)]
        train = np.concatenate(train)
        test = np.concatenate(test)
        return images, labels, train, test

def get_all_data():
    # throw out garbage labels at the end
    images1 = np.load(IMAGES_FULL)[:-130]
    labels1 = np.load(LABELS_FULL)[:-130]
    images2 = np.load(IMAGES2_FULL)
    labels2 = np.load(LABELS2_FULL)

    images = np.concatenate([images1, images2])
    labels = np.concatenate([labels1, labels2])

    count1 = len(images1)
    split1 = count1 // 10 * 9
    count2 = len(images2)
    split2 = count2 // 10 * 9

    train_ranges = [range(split1), range(count1, count1+split2)]
    test_ranges = [range(split1, count1), range(count1+split2, count1+count2)]

    return images, labels, train_ranges, test_ranges

def open(img, size=3, it=4):
    original_type = img.dtype
    img = img.astype(np.uint8)
    # kernel = np.ones((size, size), dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size, size))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=it).astype(original_type)
    # if open_first:
    #     open = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=it)
    #     out = cv2.morphologyEx(open, cv2.MORPH_CLOSE, kernel, iterations=it)
    # else:
    #     close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=it)
    #     out = cv2.morphologyEx(close, cv2.MORPH_OPEN, kernel, iterations=it)
    # return out.astype(original_type)

def test_open(i, images=IMG, labels=LAB, size=3, it=4):
    import cmd
    img = images[i]
    lab = labels[i]
    opened = open(lab, size, it)
    img_stacked = np.tile(img, (2,1,1))
    lab_stacked = np.concatenate([lab, opened])
    overlayed = cmd.overlay(img_stacked, lab_stacked)
    cmd.imshow(overlayed)
