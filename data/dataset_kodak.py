import os
import sys
import tarfile
from six.moves import urllib
import tensorflow as tf
import os
import sys
import zipfile
import requests

_HEIGHT = 512
_WIDTH = 768
_NUM_CHANNELS = 3
_NUM_IMAGES = {
    'train': 24,
    'validation': 24,
    'test': 24,
}

SHUFFLE_BUFFER = _NUM_IMAGES['train']
SHAPE = [_HEIGHT, _WIDTH, _NUM_CHANNELS]


def get_dataset(is_training, data_dir):
    """Returns a dataset object"""
    maybe_download_and_extract(data_dir)

    file_pattern = os.path.join(data_dir, "kodim*.png")
    filename_dataset = tf.data.Dataset.list_files(file_pattern)
    return filename_dataset.map(lambda x: tf.image.decode_png(tf.io.read_file(x)))


def parse_record(raw_record, _mode, dtype):
    """Parse CIFAR-10 image and label from a raw record."""
    image = tf.reshape(raw_record, [_HEIGHT, _WIDTH, _NUM_CHANNELS])
    # normalise images to range 0-1
    image = tf.cast(image, dtype)
    image = tf.divide(image, 255.0)


    return image, image


def preprocess_image(image, is_training):
    """Preprocess a single image of layout [height, width, depth]."""
    if is_training:
        # Resize the image to add four extra pixels on each side.
        image = tf.image.resize_image_with_crop_or_pad(
            image, _HEIGHT + 8, _WIDTH + 8)

        # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
        image = tf.random_crop(image, [_HEIGHT, _WIDTH, _NUM_CHANNELS])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

    # Subtract off the mean and divide by the variance of the pixels.
    image = tf.image.per_image_standardization(image)
    return image
import os
import requests
import zipfile

def maybe_download_and_extract(data_dir):
    """Download and extract the zip file from the provided Kaggle link."""
    try:
        os.makedirs(data_dir, exist_ok=True)
    except OSError:
        print(f"Creation of the directory {data_dir} failed.")
        return

    print("Downloading and extracting the dataset...")

    url = "https://drive.google.com/uc?id=1ZHginEzirPu8fqVdjvVSN-s9Gx0E6hdt"
    file_path = os.path.join(data_dir, "kodak-dataset.zip")

    # Download the zip file
    response = requests.get(url)
    with open(file_path, "wb") as f:
        f.write(response.content)

    # Extract the contents of the zip file
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    # Remove the downloaded zip file
    os.remove(file_path)

    # Check if there are 24 PNG files in the data directory
    png_files = [file for file in os.listdir(data_dir) if file.lower().endswith(".png")]
    if len(png_files) != 24:
        print(f"Expected 24 PNG files, but found {len(png_files)} files.")
        return

    print("Successfully downloaded and extracted the dataset.")


maybe_download_and_extract("/tmp/train_data")
