import os
import sys
import subprocess
import shutil
import argparse
import pp


def extract_each_class(source_dir, target_dir, class_index):
    source_class_dir = os.path.join(source_dir, class_index)
    videos = os.listdir(source_class_dir)
    videos.sort()

    target_class_dir = os.path.join(target_dir, class_index)
    if not os.path.exists(target_class_dir):
        os.makedirs(target_class_dir)

    for each_video in videos:
        source_video_name = os.path.join(source_class_dir, each_video)
        video_prefix = each_video.split('.')[0]
        target_video_frames_folder = os.path.join(target_class_dir, video_prefix)
        if not os.path.exists(target_video_frames_folder):
            os.makedirs(target_video_frames_folder)
        target_frames = os.path.join(target_video_frames_folder, 'frame_%05d.jpg')

        try:
            # change videos to 30 fps and extract video frames
            subprocess.call('ffmpeg -nostats -loglevel 0 -i %s -filter:v fps=fps=30 -s 340x256 -q:v 2 %s' %
                            (source_video_name, target_frames), shell=True)

            # sanity check video frames
            video_frames = os.listdir(target_video_frames_folder)
            video_frames.sort()

            if len(video_frames) == 300:
                # exactly 300 frames
                continue
            elif len(video_frames) > 300:
                # remove frames longer than 300
                for i in range(300, len(video_frames)):
                    os.remove(os.path.join(target_video_frames_folder, video_frames[i]))
            else:
                # duplicate videos with less than 300 frames
                last_file = 'frame_%05d.jpg' % (len(video_frames) - 1)
                last_file = os.path.join(target_video_frames_folder, last_file)
                for i in range(len(video_frames), 300 + 1):
                    new_file = 'frame_%05d.jpg' % i
                    new_file = os.path.join(target_video_frames_folder, new_file)
                    shutil.copyfile(last_file, new_file)
        except:
            print('Video %s decode failed.' % (source_video_name))
            continue


def extract_frames(source_dir, target_dir, n_cpus):
    source_classes = os.listdir(source_dir)
    source_classes.sort()
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    job_server = pp.Server(ncpus=n_cpus)
    print("Starting pp with", job_server.get_ncpus(), "workers")

    jobs = [(class_index, job_server.submit(extract_each_class, (source_dir, target_dir, class_index, ),
                                            (), ("os", "subprocess", "shutil",)))
            for class_index in source_classes]

    for class_index, job in jobs:
        result = job()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract frames of Kinetics400 dataset")
    parser.add_argument('--source_dir', type=str, help='the directory of raw videos')
    parser.add_argument('--target_dir', type=str, help='the directory which is used to store the extracted frames')
    parser.add_argument('--n_cpus', type=int, default=64, help='the number of CPUs')
    args = parser.parse_args()

    assert args.source_dir, "You must give the source_dir of raw videos!"
    assert args.target_dir, "You must give the traget_dir for storing the extracted frames!"

    import time
    tic = time.time()
    extract_frames(args.source_dir, args.target_dir, args.n_cpus)
    print(time.time() - tic)
