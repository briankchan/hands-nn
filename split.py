import numpy as np

FILES = "data/hands2-3650"
frames = range(500)


images = np.load(FILES + "-images.npy")
labels = np.load(FILES + "-labels.npy")

np.save(FILES + "-images-500.npy", images[frames])
np.save(FILES + "-labels-500.npy", labels[frames])
