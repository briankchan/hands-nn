from math import ceil
import numpy as np
import cv2

DATA1 = "data/hands1-3650"
DATA2 = "data/hands2-3650"
IMAGES_SMALL = DATA1 + "-images-500.npy"
LABELS_SMALL = DATA1 + "-labels-500.npy"

IMAGES1 = DATA1 + "-images.npy"
LABELS1 = DATA1 + "-labels.npy"

IMAGES2 = DATA2 + "-images.npy"
LABELS2 = DATA2 + "-labels.npy"

# small dataset for testing; made global for convenience
IMG, LAB, TRAIN, TEST = [None] * 4

# def split(ranges, test_chunks, num_chunks):
#     test_indices = []
#     train_indices = []
#     for indices in ranges:
#         chunk_size = ceil(len(indices) / num_chunks)
#         chunks = np.array(list(chunks(indices, chunk_size)))
#         test_indices += chunks[test_chunks].tolist()
#         train_indices += np.delete(chunks, test_chunks, axis=0).tolist()
#
#     test_indices = np.concatenate(test_indices)
#     train_indices = np.concatenate(train_indices)
#
#     return train_indices, test_indices

def split(count, use_dev=True, num_chunks=10, dev_chunks=5, test_chunks=9):
    if dev_chunks is list:
        dev_chunks = tuple(dev_chunks)
    if test_chunks is list:
        test_chunks = tuple(test_chunks)
    dev_chunks = np.r_[dev_chunks]
    test_chunks = np.r_[test_chunks]

    both = np.r_[dev_chunks, test_chunks]
    size = ceil(count / num_chunks)
    slices = np.array([slice(i, i+size) for i in range(0, count, size)])

    train = np.delete(slices, both)
    test = slices[dev_chunks if use_dev else test_chunks]
    return train, test

def get_small_dataset():
    global IMG, LAB, TRAIN, TEST
    if IMG is None:
        IMG = np.load(IMAGES_SMALL)
        LAB = np.load(LABELS_SMALL)
        TRAIN = [range(400)]
        TEST = [range(400, len(IMG))]
    return IMG, LAB, TRAIN, TEST

def get_dataset(dataset, use_dev=True, rem_noise=False, cross_validate=False):
    if dataset == "small":
        images, labels, train, test = get_small_dataset()
    elif dataset == "1":
        images = np.load(IMAGES1)[:-130]
        labels = np.load(LABELS1)[:-130]
        if cross_validate:
            train = []
            test = []
            for i in range(9):
                tr, ts = split(len(images), use_dev, dev_chunks=i)
                train.append(tr)
                test.append(ts)
        else:
            train, test = split(len(images), use_dev)
    elif dataset == "2":
        images = np.load(IMAGES2)
        labels = np.load(LABELS2)
        if cross_validate:
            train = []
            test = []
            for i in range(9):
                tr, ts = split(len(images), use_dev, dev_chunks=i)
                train.append(tr)
                test.append(ts)
        else:
            train, test = split(len(images), use_dev)
    elif dataset == "both":
        images1 = np.load(IMAGES1)[:-130]
        labels1 = np.load(LABELS1)[:-130]
        images2 = np.load(IMAGES2)
        labels2 = np.load(LABELS2)
        images = np.concatenate([images1, images2])
        labels = np.concatenate([labels1, labels2])

        if cross_validate:
            train = []
            test = []
            for i in range(9):
                tr1, ts1 = split(len(images1), use_dev, dev_chunks=i)
                tr2, ts2 = split(len(images2), use_dev, dev_chunks=i)
                train.append(np.concatenate([tr1, tr2]))
                test.append(np.concatenate([ts1, ts2]))
        else:
            train1, test1 = split(len(images1), use_dev)
            train2, test2 = split(len(images2), use_dev)
            train = np.concatenate([train1, train2])
            test = np.concatenate([test1, test2])
    else:
        raise ValueError("Invalid dataset name")

    if rem_noise:
        for i, frame in enumerate(labels):
            labels[i] = remove_noise(frame)

    return images, labels, train, test

def remove_noise(img, size=3, it=4):
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

def test_remove_noise(i, images=None, labels=None, size=3, it=4):
    import cmd
    if images is None or labels is None:
        images, labels, _, _ = get_small_dataset()
    img = images[i]
    lab = labels[i]
    opened = remove_noise(lab, size, it)
    img_stacked = np.tile(img, (2,1,1))
    lab_stacked = np.concatenate([lab, opened])
    overlayed = cmd.overlay(img_stacked, lab_stacked)
    cmd.imshow(overlayed)
