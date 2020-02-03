import os
from argparse import ArgumentParser
from time import sleep

import cv2 as cv

from wagon_tracking.videostream import VideoFileStream
from wagon_tracking.transforms import DistortionRectifier, ImageDownscaleTransform


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '-i', '--input', type=str, required=True, help='The input video'
    )
    parser.add_argument(
        '-o', '--output', type=str, required=True, help='The output video name'
    )
    parser.add_argument(
        '--max-chunks',
        type=int,
        required=False,
        help='The maximum number of video chunks to be generated from the video. If max-chunks <= 0,'
        ' the video will not be splitted.',
    )
    parser.add_argument(
        '--frames-per-chunk',
        type=int,
        required=False,
        help='The maximum number of frames in video chunks that will be generated from the video.',
    )
    parser.add_argument(
        '--frames-downscale-factor',
        type=int,
        required=False,
        help='The downscale factor of the frames dimensions',
    )
    parser.add_argument(
        '--frame-buffer-size',
        type=int,
        default=128,
        help='The size of the frame buffer used by the video stream',
    )
    parser.add_argument(
        '--fix-distortion',
        action='store_true',
        help='If set, the radial distortion of the eyefish camera will be corrected in the output video(s).',
    )
    parser.add_argument(
        '--distortion-param-file',
        type=str,
        required=False,
        help='The path to the camera/distortion parameters file.',
    )
    args = parser.parse_args()

    #####################################################################
    input_video = os.path.expanduser(args.input)

    # Load the transforms
    transforms = []
    if args.frames_downscale_factor and args.frames_downscale_factor > 1:
        transforms.append(ImageDownscaleTransform(args.frames_downscale_factor))
    if args.fix_distortion:
        if args.distortion_param_file:
            params_file = args.distortion_param_file
        else:
            params_file = './resources/new_parameters.pkl.gz'
        transforms.append(DistortionRectifier(params_file))

    # Initialize the input video streaming.
    video_stream = VideoFileStream(
        input_video, queue_sz=args.frame_buffer_size, transforms=transforms
    )
    video_fps = video_stream.get(cv.CAP_PROP_FPS)
    video_fourcc = int(video_stream.get(cv.CAP_PROP_FOURCC))
    frame_width = int(video_stream.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_stream.get(cv.CAP_PROP_FRAME_HEIGHT))
    if args.frames_downscale_factor and args.frames_downscale_factor > 1:
        new_frame_sz = (
            int(frame_width // args.frames_downscale_factor),
            int(frame_height // args.frames_downscale_factor),
        )
    else:
        new_frame_sz = (int(frame_width), int(frame_height))

    video_stream.start()
    sleep(1.0)

    #####################################################################
    output_video = os.path.expanduser(args.output)

    # Check the output folder.
    out_basedir = os.path.dirname(output_video)
    if not os.path.isdir(out_basedir):
        os.makedirs(out_basedir)

    frames_per_chunk = args.frames_per_chunk

    # If max_chunk_frames is set, get the name of the first VideoWriter.
    name_without_ext, ext = os.path.splitext(output_video)
    if frames_per_chunk and frames_per_chunk > 0:
        output_video = f'{name_without_ext}_{0}{ext}'

    max_chunks = args.max_chunks
    chunk_count = 0

    video_writer = cv.VideoWriter(output_video, video_fourcc, video_fps, new_frame_sz)

    frame_count = 0
    frames_processed = 0
    while video_stream.more():
        frame = video_stream.read()

        video_writer.write(frame)

        frame_count += 1
        frames_processed += 1

        if frames_per_chunk:
            print(f'\rchunk {chunk_count} -- Frame {frame_count}', end='')
        else:
            print(f'\rFrame {frame_count}', end='')

        if frames_per_chunk and frame_count >= frames_per_chunk:
            chunk_count += 1
            frame_count = 0

            if max_chunks and chunk_count >= max_chunks:
                break

            video_writer.release()

            output_video = f'{name_without_ext}_{chunk_count}{ext}'
            video_writer = cv.VideoWriter(
                output_video, video_fourcc, video_fps, new_frame_sz
            )

            print('')

    video_writer.release()

    print(f'\n{frames_processed} frames processed')
