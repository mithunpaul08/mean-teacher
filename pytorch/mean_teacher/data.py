"""Functions to load data from folders and augment it"""

import itertools
import logging
import os.path

from PIL import Image
import numpy as np
from torch.utils.data.sampler import Sampler


LOG = logging.getLogger('main')
NO_LABEL = -1




class RandomTranslateWithReflect:
    """Translate image randomly

    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].

    Fill the uncovered blank area with reflect padding.
    """

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))

        return new_image


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


#randomly choose propotional labeled samples per class
def relabel_dataset_RE(dataset, args):
    unlabeled_idxs = []
    labeled_ids = []
    all_labels = list(enumerate(dataset.get_labels()))   # notice enumerate also keep the indexes
    random.shuffle(all_labels)  # randomizing the relabeling
    num_classes = dataset.get_num_classes()
    num_per_classes = dataset.get_num_per_classes()
    num_labels_per_class = []

    if args.labels.isdigit():   #integer --> number of labeled datapoints
        LOG.info("[relabel dataset] Choosing " + args.labels + " NUMBER OF EXAMPLES randomly as supervision")
        num_labels = int(args.labels)
        for i in range(num_classes):
            num_c = num_per_classes[i]
            num_labels_c = int(num_labels * num_c / len(all_labels))
            num_labels_per_class.append(num_labels_c)

    else:    #float number between 0 and 100 --> percentage
        LOG.info("[relabel dataset] Choosing " + args.labels + "% OF EXAMPLES randomly as supervision")
        percent_labels = float(args.labels)
        for i in range(num_classes):
            num_c = num_per_classes[i]
            num_labels_c = int(num_c * percent_labels / 100.0)
            num_labels_per_class.append(num_labels_c)

    for idx, l in all_labels:
        if num_labels_per_class[l] > 0:
            labeled_ids.append(idx)
            num_labels_per_class[l] -= 1
        else:
            unlabeled_idxs.append(idx)
            dataset.lbl[idx] = NO_LABEL

    LOG.info("[relabel dataset] Number of LABELED examples : " + str(len(labeled_ids)))
    LOG.info("[relabel dataset] Number of UNLABELED examples : " + str(len(unlabeled_idxs)))
    LOG.info("[relabel dataset] TOTAL : " + str(len(labeled_ids)+len(unlabeled_idxs)))
    return labeled_ids, unlabeled_idxs


#randomly choose propotional labeled samples per class
def relabel_dataset_nlp(dataset, args):
    unlabeled_idxs = []
    labeled_ids = []

    all_labels = list(enumerate(dataset.get_labels()))
    random.shuffle(all_labels) # randomizing the relabeling ...
    num_classes = dataset.get_num_classes()

    if args.labels.isdigit():
        # NOTE: if it contains whole numbers --> number of labeled datapoints
        LOG.info("[relabel dataset] Choosing " + args.labels + " NUMBER OF EXAMPLES randomly as supervision")
        num_labels = int(args.labels)
    else:
        # NOTE: if it contains a float (remember even xx.00) then it is a percentage ..
        #       give a float number between 0 and 100 .. indicating percentage
        LOG.info("[relabel dataset] Choosing " + args.labels + "% OF EXAMPLES randomly as supervision")
        percent_labels = float(args.labels)
        num_labels = int(percent_labels * len(all_labels) / 100.0)

    #to make sure that the labels are evenly distributed, from each class mark x number of labels as labeled,.
    num_labels_per_cat = int(num_labels / num_classes)

    labels_hist = {}
    for _, lbl in all_labels:
        if lbl in labels_hist:
            labels_hist[lbl] += 1
        else:
            labels_hist[lbl] = 1

    num_labels_per_cat_dict = {}
    for lbl, cnt in labels_hist.items():
        num_labels_per_cat_dict[lbl] = min(labels_hist[lbl], num_labels_per_cat)

    for idx, l in all_labels:
        if num_labels_per_cat_dict[l] > 0:
            labeled_ids.append(idx)

            #reduce the count of label/category which was stored in num_labels_per_cat_dict, by 1, every time you move a label as indexed.
            num_labels_per_cat_dict[l] -= 1
        else:
            #once you run out of all the count of labels that you had earmarked for labeling in a given category, mark the rest all as unlabeled.
            unlabeled_idxs.append(idx)
            dataset.lbl[idx] = NO_LABEL

    LOG.info("[relabel dataset] Number of LABELED examples : " + str(len(labeled_ids)))
    LOG.info("[relabel dataset] Number of UNLABELED examples : " + str(len(unlabeled_idxs)))
    LOG.info("[relabel dataset] TOTAL : " + str(len(labeled_ids)+len(unlabeled_idxs)))
    return labeled_ids, unlabeled_idxs


def get_all_label_indices(dataset, args):

    #this command returns all the labels and its corresponding indices eg:[198,2]
    all_labels = list(enumerate(dataset.get_labels()))

    #note that even though the labels are shuffled up, we are keeping track/returning only the shuffled indices. so it all works out fine.
    random.shuffle(all_labels)

    #get all the indices alone
    all_indices=[]
    for idx,_  in all_labels:
        all_indices.append(idx)
    return all_indices


def relabel_dataset(dataset, labels):
    unlabeled_idxs = []
    for idx in range(len(dataset.imgs)):
        path, _ = dataset.imgs[idx]
        filename = os.path.basename(path)
        if filename in labels:
            label_idx = dataset.class_to_idx[labels[filename]]
            dataset.imgs[idx] = path, label_idx
            del labels[filename]
        else:
            dataset.imgs[idx] = path, NO_LABEL
            unlabeled_idxs.append(idx)

    if len(labels) != 0:
        message = "List of unlabeled contains {} unknown files: {}, ..."
        some_missing = ', '.join(list(labels.keys())[:5])
        raise LookupError(message.format(len(labels), some_missing))

    labeled_idxs = sorted(set(range(len(dataset.imgs))) - set(unlabeled_idxs))

    return labeled_idxs, unlabeled_idxs


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
