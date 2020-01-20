import os
from argparse import ArgumentParser
from shutil import copyfile
from glob import iglob

import numpy as np

from wagon_tracking.utils import get_realpath


def _split_dataset(anno_data, output_folder, percent_train_samples):
    n_train = int(percent_train_samples * len(anno_data))

    items = [img_key for img_key, _, _ in anno_data]

    np.random.shuffle(items)

    sets_folder = os.path.join(output_folder, 'ImageSets', 'Main')
    if not os.path.isdir(sets_folder):
        os.makedirs(sets_folder)

    with open(os.path.join(sets_folder, 'trainval.txt'), 'w+') as f:
        for img_key in items[:n_train]:
            f.write(f'{img_key}\n')

    with open(os.path.join(sets_folder, 'test.txt'), 'w+') as f:
        for img_key in items[n_train:]:
            f.write(f'{img_key}\n')


def _get_frames_keys(frames_folder, frames_ext):
    frames_template = os.path.join(frames_folder, '*.' + frames_ext)

    frames_keys = []

    for frame_path in iglob(frames_template):
        basename = os.path.basename(frame_path)
        frames_keys.append(os.path.splitext(basename)[0])

    np.random.shuffle(frames_keys)
    return frames_keys


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '-df',
        '--dataset-folder',
        type=str,
        required=True,
        help='The folder of the dataset.',
    )
    parser.add_argument(
        '-p',
        '--percent-train-samples',
        type=float,
        default=0.7,
        help='The percentage of train samples. Default is 70%%',
    )
    parser.add_argument(
        '--frames-ext',
        type=str,
        default='jpg',
        help='The extension of the input frames.',
    )
    args = parser.parse_args()

    dataset_folder = get_realpath(args.dataset_folder)

    frames_folder = os.path.join(dataset_folder, 'JPEGImages')
    assert os.path.isdir(frames_folder)

    frames_keys = _get_frames_keys(frames_folder, args.frames_ext)

    n_train = int(args.percent_train_samples * len(frames_keys))

    sets_folder = os.path.join(dataset_folder, 'ImageSets', 'Main')
    if not os.path.isdir(sets_folder):
        os.makedirs(sets_folder)

    with open(os.path.join(sets_folder, 'trainval.txt'), 'w+') as f:
        for img_key in frames_keys[:n_train]:
            f.write(f'{img_key}\n')

    with open(os.path.join(sets_folder, 'test.txt'), 'w+') as f:
        for img_key in frames_keys[n_train:]:
            f.write(f'{img_key}\n')

    copyfile('resources/labels.txt', os.path.join(dataset_folder, 'labels.txt'))
