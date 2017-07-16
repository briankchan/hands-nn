import numpy as np
import matplotlib.pyplot as plt
from cnn import CNN
import data
from data.data import split, IMG, LAB, TRAIN, TEST




def split_run_save(images, labels, train, test, *args, **kwargs):
    train, val = split(train, [5], 9)
    pred = run(images, labels, train, test, *args, **kwargs)
    m.save()
    return pred

def split_and_run(images, labels, test_chunks, num_chunks=9, ranges=None, *args, **kwargs):
    if ranges is None:
        ranges = [range(len(images))]

    train_indices, test_indices = split(ranges, test_chunks, num_chunks)

    return run(images, labels, train_indices, test_indices, *args, **kwargs), test_indices

def randsplit_and_run(images, labels, num_chunks=9, num_test_chunks=1, ranges=None, *args, **kwargs):
    test_chunks = np.random.choice(range(num_chunks), num_test_chunks, replace=False)
    return split_and_run(images, labels, test_chunks, num_chunks, ranges, *args, **kwargs)

def side_concat(img, lab):
    a = img
    b = (lab * 255).repeat(3).reshape(480, 640, 3).astype(np.uint8)
    return np.concatenate((a, b), axis=1)

def overlay(img, lab, truth=None):
    img = np.copy(img)
    img[lab, 0] = 255
    img[lab, 1:] //= 2

    if truth is not None:
        img[truth, 2] = 255

        intersection = truth*lab
        img[truth^intersection, :2] //= 2
    return img

def imshow(img):
    plt.imshow(img)
    plt.show()

def labshow(labels, i, images=IMG, truth=LAB, test_indices=TEST):
    if test_indices is None:
        test_indices = range(len(images))
    imshow(overlay(images[test_indices[i]], labels[i], truth[test_indices[i]]))

def parse_extra_args(args):
    output = {}
    for i in range(0, len(args), 2):
        arg = args[i]
        val = args[i+1]
        if arg.startswith("--"):
            arg = arg[2:]

        if val.lower() == "true":
            val = True
        elif val.lower() == "false":
            val = False
        else:
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass
        output[arg] = val
    return output

MODELS = {
    "cnn": CNN
}


def main(model, dataset, use_dev=True, save=False, load=None, path=None, **model_args):
    images, labels, train_idx, test_idx = data.get_dataset(dataset, use_dev)
    model = MODELS[model]
    if load is None:
        m = model(**model_args)
        m.train(images, labels, train_idx)
        if save:
            m.save(path)
    else:
        m = model.load(None if load == -1 else load, path)
    m.test(images, labels, test_idx)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", default="cnn")
    parser.add_argument("dataset", default=1, choices=[0, 1, 2], type=int)
    parser.add_argument("--save", action="store_true") # mutally excluse save/load path?
    parser.add_argument("--load", default=argparse.SUPPRESS, type=int)
    parser.add_argument("--path")
    args, extra = parser.parse_known_args()
    model_args = parse_extra_args(extra)
    # main(args.model, args.dataset, args.save, args.load, **model_args)
    all_args = vars(args)
    all_args.update(model_args)
    main(**all_args)
