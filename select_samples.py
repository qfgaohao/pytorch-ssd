import os
import xml.etree.ElementTree as et
from argparse import ArgumentParser
from glob import iglob
from shutil import copyfile

from wagon_tracking.utils import get_realpath


def _get_anno_name(anno_path):
    return os.path.splitext(os.path.basename(anno_path))[0]


def _has_object(annotation_path):
    obj = et.parse(annotation_path).find('object')
    return obj is not None


def get_annotations_dict(frames_folder, annotations_folder, frames_ext):
    anno_data = []

    frames_template = os.path.join(frames_folder, '{}.' + frames_ext)

    for anno_path in iglob(os.path.join(annotations_folder, '*.xml')):
        anno_name = _get_anno_name(anno_path)
        frame_path = frames_template.format(anno_name)
        if not (os.path.isfile(frame_path) and _has_object(anno_path)):
            continue
        anno_data.append((anno_name, frame_path, anno_path))

    return anno_data


def _save_database(anno_data, output_folder, frames_ext):
    frames_folder = os.path.join(output_folder, 'JPEGImages')
    if not os.path.isdir(frames_folder):
        os.makedirs(frames_folder)

    annotations_folder = os.path.join(output_folder, 'Annotations')
    if not os.path.isdir(annotations_folder):
        os.makedirs(annotations_folder)

    frames_template = os.path.join(frames_folder, '{}.' + frames_ext)
    anno_template = os.path.join(annotations_folder, '{}.xml')

    for name, frame_path, anno_path in anno_data:
        copyfile(frame_path, frames_template.format(name))
        copyfile(anno_path, anno_template.format(name))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '-ia',
        '--input-annos',
        type=str,
        required=True,
        help='The folder where all the original annotations are',
    )
    parser.add_argument(
        '-if',
        '--input-frames',
        type=str,
        required=True,
        help='The folder where all the original frames are',
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        required=True,
        help='The output folder of the dataset.',
    )
    parser.add_argument(
        '--frames-ext',
        type=str,
        default='jpg',
        help='The extension of the input frames.',
    )
    args = parser.parse_args()

    input_annos_folder = get_realpath(args.input_annos)
    input_frames_folder = get_realpath(args.input_frames)

    output_folder = get_realpath(args.output)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    anno_data = get_annotations_dict(
        input_frames_folder, input_annos_folder, args.frames_ext
    )

    _save_database(
        anno_data, output_folder, args.frames_ext
    )
