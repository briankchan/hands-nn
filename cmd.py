import importlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.widgets import Slider, CheckButtons

import data

IMAGES, LABELS, TRAIN, TEST = [None] * 4

def side_concat(img, lab):
    a = img
    b = (lab * 255).repeat(3).reshape(480, 640, 3).astype(np.uint8)
    return np.concatenate((a, b), axis=1)

def overlay(img, lab=None, truth=None):
    if lab is None:
        if truth is None:
            return img
        img = np.copy(img)
        img[truth, 2] = 255
        img[truth, :2] //= 2
        return img
    else:
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

def vidshow(func, max, start=0):
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    img_ax = plt.imshow(func(start))

    time_ax = plt.axes([0.25, 0.1, 0.65, 0.03])

    time_sld = Slider(time_ax, "Frame", 0, max - .01, valinit=start, valfmt="%d")

    def update(i):
        i = int(i)
        img_ax.set_data(func(i))
        fig.canvas.draw_idle()

    def onkeypress(event):
        if event.key == "left":
            time_sld.set_val(max(time_sld.val - 1, time_sld.valmin))
        elif event.key == "right":
            time_sld.set_val(min(time_sld.val + 1, time_sld.valmax))
        elif event.key == "ctrl+left":
            time_sld.set_val(max(time_sld.val - 30, time_sld.valmin))
        elif event.key == "ctrl+right":
            time_sld.set_val(min(time_sld.val + 30, time_sld.valmax))
        elif event.key == "home":
            time_sld.set_val(time_sld.valmin)
        elif event.key == "end":
            time_sld.set_val(time_sld.valmax)

    time_sld.on_changed(update)
    fig.canvas.mpl_connect("key_press_event", onkeypress)
    plt.show()

def compare_runs(i1, i2, model, dataset, remove_noise=False, model2=None, dataset2=None, remove_noise2=None):
    if model2 is None:
        model2 = model
    if dataset2 is None:
        dataset2 = dataset
    if remove_noise2 is None:
        remove_noise2 = remove_noise
    pred1, _ = main(model, dataset, remove_noise=remove_noise, load=i1, show=False, bell=False)
    images1 = IMAGES
    truth1 = LABELS
    test_indices1 = np.r_[tuple(TEST)]
    pred2, _ = main(model2, dataset2, remove_noise=remove_noise2, load=i2, show=False, bell=False)
    images2 = IMAGES
    truth2 = LABELS
    test_indices2 = np.r_[tuple(TEST)]
    vidshow(lambda i:
            np.concatenate([
                overlay(images1[test_indices1[i]], pred1[i], truth1[test_indices1[i]]),
                overlay(images2[test_indices2[i]], pred2[i], truth2[test_indices2[i]])
            ], axis=1),
            len(test_indices1))

GRID_SIZE = 9
def labshow(labels, i=0, images=None, truth=None, test_indices=None):
    if images is None:
        images = IMAGES
    if truth is None:
        truth = LABELS
    if test_indices is None:
        test_indices = TEST
    test_indices = np.r_[tuple(test_indices)]

    gs = gridspec.GridSpec(GRID_SIZE, GRID_SIZE)
    fig = plt.figure()
    main_ax = fig.add_subplot(gs[:-1, 1:])
    fig.tight_layout()
    img_ax = main_ax.imshow(overlay(images[test_indices[i]], labels[i], truth[test_indices[i]]))

    time_ax = fig.add_subplot(gs[-1, 1:-1])
    time_sld = Slider(time_ax, "Frame", 0, len(test_indices) - .01, valinit=i, valfmt="%d")

    half = GRID_SIZE // 2
    selector_ax = fig.add_subplot(gs[half-2:half, 0])
    selector_chk = CheckButtons(selector_ax, ["Pred", "Truth"], [True, True])

    def update(_):
        i = int(time_sld.val)
        selected = [l1.get_visible() for l1, l2 in selector_chk.lines]
        # checkbuttons.get_status() is supposed to exist, but doesn't?
        img_ax.set_data(overlay(
            images[test_indices[i]],
            labels[i] if selected[0] else None,
            truth[test_indices[i]] if selected[1] else None))
        fig.canvas.draw_idle()
    time_sld.on_changed(update)
    selector_chk.on_clicked(update)

    def onkeypress(event):
        if event.key == "left":
            time_sld.set_val(max(time_sld.val - 1, time_sld.valmin))
        elif event.key == "right":
            time_sld.set_val(min(time_sld.val + 1, time_sld.valmax))
        elif event.key == "ctrl+left":
            time_sld.set_val(max(time_sld.val - 30, time_sld.valmin))
        elif event.key == "ctrl+right":
            time_sld.set_val(min(time_sld.val + 30, time_sld.valmax))
        elif event.key == "home":
            time_sld.set_val(time_sld.valmin)
        elif event.key == "end":
            time_sld.set_val(time_sld.valmax)
    fig.canvas.mpl_connect("key_press_event", onkeypress)

    plt.show()

def parse_extra_args(args):
    output = {}
    for i in range(0, len(args), 2):
        arg = args[i]
        val = args[i+1]
        if arg.startswith("--"):
            arg = arg[2:]
        arg = arg.replace("-", "_")

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
         bell=True,
         **model_args):
    global IMAGES, LABELS, TRAIN, TEST
    try:
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
            accuracy, precision, recall, f1 = [], [], [], []
            conf_mat = np.zeros([2, 2], dtype=np.int)
            num_folds = len(TRAIN)
            for i, (train, test) in enumerate(zip(TRAIN, TEST), 1):
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
                labshow(pred, images=IMAGES, truth=LABELS, test_indices=TEST)
            return pred, stats
    finally:
        if bell:
            print("\a")


def parse_args(args=None):
    import argparse

    class CustomStoreFalse(argparse._StoreFalseAction):
        def __init__(self, dest=None, **kw):
            super().__init__(dest=dest[3:], **kw)

    parser = argparse.ArgumentParser()
    parser.add_argument("model", default="cnn")
    parser.add_argument("dataset", default="small")
    parser.add_argument("--remove-noise", action="store_true")
    parser.add_argument("--save", action="store_true")  # TODO mutually exclude save/load path?
    parser.add_argument("--load", default=argparse.SUPPRESS, type=int)
    parser.add_argument("--path")
    parser.add_argument("--cross-validate", action="store_true")
    parser.add_argument("--no-show", action=CustomStoreFalse)
    parser.add_argument("--no-bell", action=CustomStoreFalse)
    args, extra = parser.parse_known_args(args)
    model_args = parse_extra_args(extra)
    all_args = vars(args)
    all_args.update(model_args)
    return all_args

if __name__ == "__main__":
    main(**parse_args())
