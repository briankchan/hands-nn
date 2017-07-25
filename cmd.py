import importlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import data
# from data.data import split

IMAGES, LABELS, TRAIN, TEST = [None] * 4

# def split_run_save(images, labels, train, test, *args, **kwargs):
#     train, val = split(train, [5], 9)
#     pred = run(images, labels, train, test, *args, **kwargs)
#     m.save()
#     return pred
#
# def split_and_run(images, labels, test_chunks, num_chunks=9, ranges=None, *args, **kwargs):
#     if ranges is None:
#         ranges = [range(len(images))]
#
#     train_indices, test_indices = split(ranges, test_chunks, num_chunks)
#
#     return run(images, labels, train_indices, test_indices, *args, **kwargs), test_indices
#
# def randsplit_and_run(images, labels, num_chunks=9, num_test_chunks=1, ranges=None, *args, **kwargs):
#     test_chunks = np.random.choice(range(num_chunks), num_test_chunks, replace=False)
#     return split_and_run(images, labels, test_chunks, num_chunks, ranges, *args, **kwargs)

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

def labshow(labels, i=0, images=None, truth=None, test_indices=None):
    if images is None:
        images = IMAGES
    if truth is None:
        truth = LABELS
    if test_indices is None:
        test_indices = TEST
    imshow(overlay(images[test_indices[i]], labels[i], truth[test_indices[i]]))

def interactive_labshow(labels, i=0, images=None, truth=None, test_indices=None):
    if images is None:
        images = IMAGES
    if truth is None:
        truth = LABELS
    if test_indices is None:
        test_indices = TEST
    test_indices = np.r_[tuple(test_indices)]

    fig, ax = plt.subplots()
    plt.subplots_adjust(left = 0.25, bottom = 0.25)
    img_ax = plt.imshow(overlay(images[test_indices[i]], labels[i], truth[test_indices[i]]))

    time_ax = plt.axes([0.25, 0.1, 0.65, 0.03])

    time_sld = Slider(time_ax, "Frame", 0, len(test_indices) - .01, valinit=i, valfmt="%d")

    def update(i):
        i = int(i)
        img_ax.set_data(overlay(images[test_indices[i]], labels[i], truth[test_indices[i]]))
        fig.canvas.draw_idle()

    def onkeypress(event):
        if event.key == "left":
            time_sld.set_val(max(time_sld.val - 1, time_sld.valmin))
        elif event.key == "right":
            time_sld.set_val(min(time_sld.val + 1, time_sld.valmax))
        elif event.key == "home":
            time_sld.set_val(time_sld.valmin)
        elif event.key == "end":
            time_sld.set_val(time_sld.valmax)

    time_sld.on_changed(update)
    fig.canvas.mpl_connect("key_press_event", onkeypress)
    plt.show()

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
    "cnn": ("cnn", "CNN"),
    "cnn_inception": ("cnn_inception", "CNNInception")
}

def import_model(model):
    module_name, class_name = MODELS[model]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def main(model, dataset,
         use_dev=True,
         remove_noise=False,
         save=False,
         load=None,
         path=None,
         cross_validate=False,
         show=True,
         **model_args):
    global IMAGES, LABELS, TRAIN, TEST
    print("Loading data")
    IMAGES, LABELS, TRAIN, TEST = data.get_dataset(
        dataset,
        use_dev=use_dev,
        rem_noise=remove_noise,
        cross_validate=cross_validate)
    model = import_model(model)

    if cross_validate:
        print("Creating model")
        m = model(**model_args)
        m._class_log_path_pattern = "cnn_crossval/set1/run{}"  # TODO TESTING
        accuracy, precision, recall, f1 = [], [], [], []
        conf_mat = np.zeros([2, 2], dtype=np.int)
        num_folds = len(TRAIN)
        for i, (train, test) in enumerate(zip(TRAIN, TEST)):
            m.reset()
            print("Training model {}/{}".format(i, num_folds))
            m.train(IMAGES, LABELS, train)
            if save:
                print("Saving run {}".format(m.run_num))
                m.save(path)

            print("Testing model {}/{}".format(i, num_folds))
            _, results = m.test(IMAGES, LABELS, test)

            accuracy.append(results["accuracy"])
            precision.append(results["precision"])
            recall.append(results["recall"])
            f1.append(results["f1"])
            conf_mat += results["confusion_matrix"]

        print("Avg Accuracy:", sum(accuracy)/num_folds)
        print("Avg Precision:", sum(precision)/num_folds)
        print("Avg Recall:", sum(recall)/num_folds)
        print("Avg F1:", sum(f1)/num_folds)
        print("Overall Confusion Matrix")
        print(conf_mat)
    else:
        if load is None:
            print("Creating model")
            m = model(**model_args)
            print("Training model")
            m.train(IMAGES, LABELS, TRAIN)
            if save:
                print("Saving run {}".format(m.run_num))
                m.save(path)
        else:
            if load == -1:
                print("Loading run ", end="")
                m = model.load(None, path)
                print(m.run_num)
            else:
                print("Loading run {}".format(load))
                m = model.load(load, path)
        print("Testing model")
        pred, stats = m.test(IMAGES, LABELS, TEST)
        if show:
            interactive_labshow(pred, images=IMAGES, truth=LABELS, test_indices=TEST)
        return pred, stats


def parse_args(args=None):
    import argparse

    class CustomStoreFalse(argparse._StoreFalseAction):
        def __init__(self, dest=None, **kw):
            super().__init__(dest=dest[3:], **kw)

    parser = argparse.ArgumentParser()
    parser.add_argument("model", default="cnn")
    parser.add_argument("dataset", default="small")
    parser.add_argument("--remove-noise", action="store_true")
    parser.add_argument("--save", action="store_true") # mutally exclude save/load path?
    parser.add_argument("--load", default=argparse.SUPPRESS, type=int)
    parser.add_argument("--path")
    parser.add_argument("--cross-validate", action="store_true")
    parser.add_argument("--no-show", action=CustomStoreFalse)
    args, extra = parser.parse_known_args(args)
    model_args = parse_extra_args(extra)
    all_args = vars(args)
    all_args.update(model_args)
    return all_args

if __name__ == "__main__":
    main(**parse_args())
