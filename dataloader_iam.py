# In this file we load our dataset which we create in lmdb file. we open it in readonly mode and create our file words.txt and move everything
# to it. and we can take measuremnets according to different functions. 

import pickle # Pickle in Python is primarily used in serializing and deserializing a Python object structure.
import random # used to generate random numbers
import cv2 # cv2 is the module import name for opencv-python.
import lmdb # This is a universal Python binding for the LMDB 'Lightning' Database
import numpy as np # used for working with arrays
from path import Path # provides various classes representing file system paths with semantics appropriate for different operating systems
from collections import namedtuple # allows you to create tuple subclasses with named fields
from typing import Tuple # it lets you specify a specific number of elements expected and the type of each position

# create classes
Sample = namedtuple('Sample', 'gt_text, file_path') # create tuple named 'Sample' with properties 'gt_text, file_path'
Batch = namedtuple('Batch', 'imgs, gt_texts, batch_size')

class DataLoaderIAM:
    def __init__(self,
                 data_dir: Path,
                 batch_size: int,
                 data_split: float = 0.95,
                 fast: bool = True) -> None:
        """Loader for dataset."""

        assert data_dir.exists() # The assert keyword lets you test if a condition in your code returns True, if not, the program will raise an AssertionError.

        self.fast = fast
        if fast:
            self.env = lmdb.open(str(data_dir / 'lmdb'), readonly=True) # open databse in readonly mode

        self.data_augmentation = False
        self.curr_idx = 0
        self.batch_size = batch_size
        self.samples = []

        f = open(data_dir / 'gt/words.txt')
        chars = set()
        bad_samples_reference = ['a01-117-05-02', 'r06-022-03-05']  # known broken images in IAM dataset
        for line in f:
            # ignore comment line
            if not line or line[0] == '#':
                continue

            line_split = line.strip().split(' ')
            assert len(line_split) >= 9

            # filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
            file_name_split = line_split[0].split('-')
            file_name_subdir1 = file_name_split[0]
            file_name_subdir2 = f'{file_name_split[0]}-{file_name_split[1]}'
            file_base_name = line_split[0] + '.png'
            file_name = data_dir / 'img' / file_name_subdir1 / file_name_subdir2 / file_base_name

            if line_split[0] in bad_samples_reference:
                print('Ignoring known broken image:', file_name)
                continue

            # GT text are columns starting at 9
            gt_text = ' '.join(line_split[8:])
            chars = chars.union(set(list(gt_text)))

            # put sample into list
            self.samples.append(Sample(gt_text, file_name))

        # split into training and validation set: 95% - 5%
        split_idx = int(data_split * len(self.samples))
        self.train_samples = self.samples[:split_idx]
        self.validation_samples = self.samples[split_idx:]

        # put words into lists
        self.train_words = [x.gt_text for x in self.train_samples]
        self.validation_words = [x.gt_text for x in self.validation_samples]

        # start with train set
        self.train_set()

        # list of all chars in dataset
        self.char_list = sorted(list(chars))

    def train_set(self) -> None:
        """Switch to randomly chosen subset of training set."""
        self.data_augmentation = True
        self.curr_idx = 0
        random.shuffle(self.train_samples)
        self.samples = self.train_samples
        self.curr_set = 'train'

    def validation_set(self) -> None:
        """Switch to validation set."""
        self.data_augmentation = False
        self.curr_idx = 0
        self.samples = self.validation_samples
        self.curr_set = 'val'

    def get_iterator_info(self) -> Tuple[int, int]:
        """Current batch index and overall number of batches."""
        if self.curr_set == 'train':
            num_batches = int(np.floor(len(self.samples) / self.batch_size))  # type: ignore # train set: only full-sized batches
        else:
            num_batches = int(np.ceil(len(self.samples) / self.batch_size))  # type: ignore # val set: allow last batch to be smaller
        curr_batch = self.curr_idx // self.batch_size + 1
        return curr_batch, num_batches

    def has_next(self) -> bool:
        """Is there a next element?"""
        if self.curr_set == 'train':
            return self.curr_idx + self.batch_size <= len(self.samples)  # train set: only full-sized batches
        else:
            return self.curr_idx < len(self.samples)  # val set: allow last batch to be smaller

    def _get_img(self, i: int) -> np.ndarray:
        """get an image"""
        if self.fast:
            with self.env.begin() as txn:
                basename = Path(self.samples[i].file_path).basename()
                data = txn.get(basename.encode("ascii"))
                img = pickle.loads(data)
        else:
            img = cv2.imread(self.samples[i].file_path, cv2.IMREAD_GRAYSCALE)
        return img

    def get_next(self) -> Batch:
        """Get next element."""
        batch_range = range(self.curr_idx, min(self.curr_idx + self.batch_size, len(self.samples)))
        imgs = [self._get_img(i) for i in batch_range]
        gt_texts = [self.samples[i].gt_text for i in batch_range]
        self.curr_idx += self.batch_size
        return Batch(imgs, gt_texts, len(imgs))
