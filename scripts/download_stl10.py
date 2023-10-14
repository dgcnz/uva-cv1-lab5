# Use the following code if necessary
# referenced code: https://cs.stanford.edu/~acoates/stl10/

from __future__ import print_function

import errno
import os
import sys
import tarfile

import matplotlib.pyplot as plt
import numpy as np

if sys.version_info >= (3, 0, 0):
    import urllib.request as urllib
else:
    import urllib

try:
    from imageio import imsave
except ImportError:
    from scipy.misc import imsave

print(sys.version_info)

# image shape
HEIGHT = 96
WIDTH = 96
DEPTH = 3

# size of a single image in bytes
SIZE = HEIGHT * WIDTH * DEPTH

# path to the directory with the data
DATA_DIR = "./data"

# url of the binary data
DATA_URL = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"

# path to the binary train file with image data
DATA_PATH_TRAIN = "./data/stl10_binary/train_X.bin"

# path to the binary train file with labels
LABEL_PATH_TRAIN = "./data/stl10_binary/train_y.bin"

# path to the binary train file with image data
DATA_PATH_TEST = "./data/stl10_binary/test_X.bin"

# path to the binary train file with labels
LABEL_PATH_TEST = "./data/stl10_binary/test_y.bin"


def read_labels(path_to_labels):
    """
    :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
    :return: an array containing the labels
    """
    with open(path_to_labels, "rb") as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels


def read_all_images(path_to_data):
    """
    :param path_to_data: the file containing the binary images from the STL-10 dataset
    :return: an array containing all the images
    """

    with open(path_to_data, "rb") as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, 3, 96, 96))

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        images = np.transpose(images, (0, 3, 2, 1))
        return images


def read_single_image(image_file):
    """
    CAREFUL! - this method uses a file as input instead of the path - so the
    position of the reader will be remembered outside of context of this method.
    :param image_file: the open file containing the images
    :return: a single image
    """
    # read a single image, count determines the number of uint8's to read
    image = np.fromfile(image_file, dtype=np.uint8, count=SIZE)
    # force into image matrix
    image = np.reshape(image, (3, 96, 96))
    # transpose to standard format
    # You might want to comment this line or reverse the shuffle
    # if you will use a learning algorithm like CNN, since they like
    # their channels separated.
    image = np.transpose(image, (2, 1, 0))
    return image


def plot_image(image):
    """
    :param image: the image to be plotted in a 3-D matrix format
    :return: None
    """
    plt.imshow(image)
    plt.show()


def save_image(image, name):
    imsave("%s.png" % name, image, format="png")


def download_and_extract():
    """
    Download and extract the STL-10 dataset
    :return: None
    """
    dest_directory = DATA_DIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split("/")[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):

        def _progress(count, block_size, total_size):
            sys.stdout.write(
                "\rDownloading %s %.2f%%"
                % (filename, float(count * block_size) / float(total_size) * 100.0)
            )
            sys.stdout.flush()

        filepath, _ = urllib.urlretrieve(DATA_URL, filepath, reporthook=_progress)
        print("Downloaded", filename)
        tarfile.open(filepath, "r:gz").extractall(dest_directory)


def save_images(images, labels, split="train"):
    print("Saving images to disk")
    i = 0
    for image in images:
        label = labels[i]
        directory = f"{DATA_DIR}/stl10/{split}/" + str(label) + "/"
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                pass
        filename = directory + str(i)
        print(filename)
        save_image(image, filename)
        i = i + 1


if __name__ == "__main__":
    # download data if needed
    download_and_extract()

    train_images = read_all_images(DATA_PATH_TRAIN)
    train_labels = read_labels(LABEL_PATH_TRAIN)

    save_images(train_images, train_labels)

    test_images = read_all_images(DATA_PATH_TEST)
    test_labels = read_labels(LABEL_PATH_TEST)

    # save images to disk
    save_images(test_images, test_labels, split="test")
