import os
import download
import glob
# Directory where you want to download and save the data-set.
# Set this before you start calling any of the functions below.
data_dir = "data/IMDB/"

# URL for the data-set on the internet.
data_url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

# Private helper-functions.

def _read_text_file(path):
    """
    Read and return all the contents of the text-file with the given path.
    It is returned as a single string where all lines are concatenated.
    """
    with open(path, 'rt', encoding='utf-8') as file:
        # Read a list of strings.
        lines = file.readlines()

        # Concatenate to a single string.
        text = " ".join(lines)

    return text

# Public functions that you may call to download the data-set from
# the internet and load the data into memory.
def maybe_download_and_extract():
    """
    Download and extract the IMDB Review data-set if it doesn't already exist
    in data_dir (set this variable first to the desired directory).
    """
    download.maybe_download_and_extract(url=data_url, download_dir=data_dir)

def load_data(train=True):
    """
    Load all the data from the IMDB Review data-set for sentiment analysis.
    :param train: Boolean whether to load the training-set (True)
                  or the test-set (False).
    :return:      A list of all the reviews as text-strings,
                  and a list of the corresponding sentiments
                  where 1.0 is positive and 0.0 is negative.
    """

    # Part of the path-name for either training or test-set.
    train_test_path = "train" if train else "test"

    # Base-directory where the extracted data is located.
    dir_base = os.path.join(data_dir, "aclImdb", train_test_path)

    # Filename-patterns for the data-files.
    path_pattern_pos = os.path.join(dir_base, "pos", "*.txt")
    path_pattern_neg = os.path.join(dir_base, "neg", "*.txt")

    # Get lists of all the file-paths for the data.
    paths_pos = glob.glob(path_pattern_pos)
    paths_neg = glob.glob(path_pattern_neg)

    # Read all the text-files.
    data_pos = [_read_text_file(path) for path in paths_pos]
    data_neg = [_read_text_file(path) for path in paths_neg]

    # Concatenate the positive and negative data.
    x = data_pos + data_neg

    # Create a list of the sentiments for the text-data.
    # 1.0 is a positive sentiment, 0.0 is a negative sentiment.
    y = [1.0] * len(data_pos) + [0.0] * len(data_neg)

    return x, y
