import numpy as np

FILES = "data/hands2-3650"
frames = range(500)


IMAGES = np.load(FILES + "-images.npy")
LABELS = np.load(FILES + "-labels.npy")

np.save(FILES + "-images-500.npy", IMAGES[frames])
np.save(FILES + "-labels-500.npy", LABELS[frames])
