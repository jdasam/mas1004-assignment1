"""Turn a folder of images into arrays. YOU write the bodies of these functions.

Your images are all different sizes, different shapes, and some of them are not
even really images. These two functions are where that mess becomes two clean
arrays that the model can eat.

Run `pytest tests/test_data.py` after you fill them in.
"""

import numpy as np

# Files with any other extension should be ignored.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_folder(root, image_size=32, color=True):
    """Read every image under `root` and return them as one array.

    `root` is a folder that holds one sub-folder per class, like this:

        data/clean/
            mug/        0001.jpg 0002.jpg ...
            wine_glass/ 0001.jpg 0002.jpg ...
            paper_cup/  0001.jpg 0002.jpg ...

    Arguments
        root        str or Path, the folder above
        image_size  int, the width and height to resize every image to
        color       bool. True keeps 3 channels (RGB), False makes it grayscale

    Returns a tuple (X, y, class_names, paths)
        X            np.float32, shape (N, D), every value between 0.0 and 1.0
                     D is image_size * image_size * 3 when color is True,
                     and image_size * image_size when color is False.
                     Each row is one image flattened into a single line of
                     numbers, channels last: pixel(0,0)R, pixel(0,0)G,
                     pixel(0,0)B, pixel(0,1)R, and so on.
        y            np.int64, shape (N,), the class index of each row
        class_names  list of str, sorted alphabetically. class_names[y[i]] is
                     the name of the class of row i.
        paths        list of Path, length N, where each row came from. Keep the
                     order the same as X and y.

    Things that will happen to you
        Some files are broken and raise an exception when you open them. Skip
        them instead of crashing.
        Some images are PNGs with transparency, or grayscale already. Convert
        everything to the same mode before you resize. Use PIL's
        .convert("RGB") when color is True and .convert("L") when it is
        False. The web page uses the same two conversions, so if you invent
        your own way of making an image gray, the browser will disagree
        with Python and your self test will fail.
        The order of files on disk is not guaranteed. Sort them so that you get
        the same result every time you run.
    """
    raise NotImplementedError("Problem 2: fill in load_folder")


def split_train_test(X, y, paths, test_ratio=0.2, seed=0):
    """Split the rows into a training part and a test part.

    Arguments
        X, y, paths  exactly what load_folder returned
        test_ratio   float between 0 and 1, the share that goes to the test side
        seed         int, so that you get the same split every time you run

    Returns (X_train, y_train, paths_train, X_test, y_test, paths_test)

    Two rules the tests check
        No image may appear on both sides. Every path belongs to exactly one.
        Every class must appear on both sides. If you shuffle badly you can end
        up with a class that has no test images at all, and then your accuracy
        number means nothing.
    """
    raise NotImplementedError("Problem 2: fill in split_train_test")
