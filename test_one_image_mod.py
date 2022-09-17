#!/usr/bin/env python
import os
import sys
import cv2
import numpy as np
import datetime
import time
from os import listdir, mkdir
from os.path import isfile, join
from subprocess import Popen, PIPE
from shutil import rmtree

from skimage.morphology import skeletonize

CURRENT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".")
sys.path.append(os.path.join(CURRENT_DIR, "../../"))
try:
    from rail_marking.segmentation.deploy import RailtrackSegmentationHandler
    from cfg import BiSeNetV2Config
except Exception as e:
    print(e)
    sys.exit(0)


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-snapshot", type=str, required=True)
    parser.add_argument("-image_path", type=str, default="")
    parser.add_argument("-video_path", type=str, default="")
    parser.add_argument("-folder_path", type=str, default="")
    parser.add_argument("-mask_path", type=str, default="")
    parser.add_argument("-no_delete", type=bool, default=False)
    parser.add_argument("-output_image_path", type=str, default="result.png")
    parser.add_argument("-num_test", type=int, default=1)

    args = parser.parse_args()

    return args


def pos_process_image(image, mask):
    # Ok, so this is fine for prototyping. But I will need to put that into c++.
    # The floodfill takes 3.5s in python, about 2ms in C++

    # Remove all rail elements that are not connected to a between rails
    height, width = mask.shape
    overmask = np.zeros((height + 2, width + 2), np.uint8)
    mask = mask.astype(np.uint8)
    mask_copy = np.array(mask, copy=True)
    rail_only = np.zeros((height, width), np.uint8)
    print(f"Taille: {height}/{width}")
    curr_start = datetime.datetime.now()

    for x in range(1, width):
        for y in range(0, height):
            if mask[y, x] == 2:
                continue
            if mask[y, x-1] == 0 and mask[y, x] == 1:
                cv2.floodFill(mask_copy, overmask, (x-1, y), 100)
                cv2.floodFill(mask_copy, overmask, (x, y), 200)
            elif mask[y, x-1] == 1 and mask[y, x] == 0:
                cv2.floodFill(mask_copy, overmask, (x, y), 100)
                cv2.floodFill(mask_copy, overmask, (x - 1, y), 200)

    for x in range(1, width):
        for y in range(0, height):
            if mask_copy[y, x] == 100:  # An image with only the rail
                rail_only[y, x] = 1

    skeleton = skeletonize(rail_only)
    skeleton = skeleton.astype(np.uint8)
    skeleton *= 200

    curr_processing_time = datetime.datetime.now() - curr_start
    print(f"processing time one frame {curr_processing_time.total_seconds() * 1000:.2f}[ms]")

    cv2.imshow("mask", mask_copy)
    cv2.imshow("skeleton", skeleton)
    cv2.imshow("Rails", rail_only)
    cv2.imshow("image", image)
    cv2.waitKey()

    pass


def process_several_images(args):
    
    segmentation_handler = RailtrackSegmentationHandler(args.snapshot, BiSeNetV2Config())

    path = args.folder_path
    list_images = [f for f in listdir(path) if isfile(join(path, f))]
    list_images.sort()
    
    start = datetime.datetime.now()
    for image in list_images:
        print(image)
        curr_start = datetime.datetime.now()
        curr_path = path+image
        img = cv2.imread(curr_path)
        mask, overlay = segmentation_handler.run(img, only_mask=False)
        total_processing_time = datetime.datetime.now() - start
        curr_processing_time = datetime.datetime.now() - curr_start
        print(f"processing time one frame {curr_processing_time.total_seconds() * 1000:.2f}[ms]")
        print(f"processing time all frame {total_processing_time.total_seconds() * 1000:.2f}[ms]")

        if args.mask_path == "":
            cv2.imshow("result", overlay)
            cv2.waitKey(20)
        else:
            mask = cv2.resize(mask, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = mask.astype(np.uint8)
            cv2.imwrite(args.mask_path+image, mask)


def process_single_image(args):

    segmentation_handler = RailtrackSegmentationHandler(args.snapshot, BiSeNetV2Config())
    image = cv2.imread(args.image_path)

    start = datetime.datetime.now()
    for i in range(args.num_test):
        mask, overlay = segmentation_handler.run(image, only_mask=False)
        # pos_process_image(image, mask)
    _processing_time = datetime.datetime.now() - start

    print("processing time one frame {}[ms]".format(_processing_time.total_seconds() * 1000 / args.num_test))
    print(overlay.shape)
    print(mask.shape)
    print(mask.dtype)
    mask = cv2.resize(mask, dsize=(overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask = mask.astype(np.uint8)
    # mask *= 100
    # cv2.imshow("mask", mask)
    # cv2.imshow("result", overlay)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite(args.output_image_path, overlay)
    cv2.imwrite("/home/yohan/Documents/data/siemens/pictures/20220712_073943_overlay.jpg", overlay)
    cv2.imwrite("/home/yohan/Documents/data/siemens/pictures/20220712_073943_mask.png", mask)


def process_video(args):
    # First extract the frames from the video in a custom folder
    # Then process the frame one by one and save the results somewhere.
    # Then create the output video.

    segmentation_handler = RailtrackSegmentationHandler(args.snapshot, BiSeNetV2Config())
    out_video_name = f"{args.video_path[:-4]}_overlay{args.video_path[-4:]}"

    temp_image_dir = f"/tmp/images_{int(time.time())}/"
    temp_overlay_dir = f"/tmp/overlay_{int(time.time())}/"
    mkdir(temp_image_dir)
    mkdir(temp_overlay_dir)

    # Split the video into a series of images.
    process_call = ["ffmpeg", "-i", args.video_path, f"{temp_image_dir}images%5d.png"]
    process = Popen(process_call, stdout=PIPE, stderr=PIPE)
    process.wait()

    # Process all the images
    list_images = [f for f in listdir(temp_image_dir) if isfile(join(temp_image_dir, f))]
    list_images.sort()
    print(f"Number of images to process: {len(list_images)}")

    start = datetime.datetime.now()
    for i, image_name in enumerate(list_images):
        if i % 25 == 24:
            print(f"[{i+1}/{len(list_images)}]")
        curr_path = temp_image_dir + image_name
        image = cv2.imread(curr_path)
        _, overlay = segmentation_handler.run(image, only_mask=False)
        # print(f"{temp_overlay_dir}{image}")
        cv2.imwrite(f"{temp_overlay_dir}{image_name}", overlay)

    process_call = ["ffmpeg", "-framerate", "30", "-pattern_type", "glob", "-i", f"{temp_overlay_dir}*.png",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", out_video_name]
    process = Popen(process_call, stdout=PIPE, stderr=PIPE)
    process.wait()

    # Clean the dir
    if not args.no_delete:
        rmtree(temp_image_dir)
        rmtree(temp_overlay_dir)


def main():
    args = get_args()
    
    if args.image_path != "":
        process_single_image(args)
    elif args.folder_path != "":
        process_several_images(args)
    elif args.video_path != "":
        process_video(args)
    else:
        print("Neither image_path or folder_path were given")


if __name__ == "__main__":
    main()

# -snapshot bisenetv2_checkpoint_BiSeNetV2_epoch_300.pth -video_path ./download/train_02_.mp4
# -snapshot bisenetv2_checkpoint_BiSeNetV2_epoch_300.pth -image_path ./download/temp/images00051.png
# -snapshot bisenetv2_checkpoint_BiSeNetV2_epoch_300.pth -image_path /home/yohan/Documents/data/siemens/pictures/20220712_073943.jpg

# -snapshot bisenetv2_checkpoint_BiSeNetV2_epoch_300.pth -folder_path /home/yohan/Documents/data/siemens/videos_with_sensordata/2022_08_17_14_00_27/frames/
# -mask_path /home/yohan/Documents/data/siemens/videos_with_sensordata/2022_08_17_14_00_27/masks/

# -snapshot bisenetv2_checkpoint_BiSeNetV2_epoch_300.pth -folder_path /home/yohan/Documents/data/download/frames/
# -mask_path /home/yohan/Documents/data/download/masks/
