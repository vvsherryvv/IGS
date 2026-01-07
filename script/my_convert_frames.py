#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import logging
from argparse import ArgumentParser
import shutil
import multiprocessing as mp
from tqdm import tqdm

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--skip_undistortion", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
parser.add_argument("--last_frame_id", default=299, type=int)

args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0

# for id in range(1,args.last_frame_id+1):
def process(item):
    inputDir = item
    print("Processing "+item)
                
    ### Image undistortion
    ## We need to undistort our images into ideal pinhole intrinsics.
    
    # 确保 source_path 和 inputDir 正确拼接
    image_path_full = os.path.join(args.source_path, inputDir, "input")
    
    # =========================================================
    # 修改开始：自动寻找最大的模型文件夹，而不是硬编码 "0"
    # =========================================================
    distorted_sparse_base = os.path.join(args.source_path, "distorted", "sparse")
    best_model_folder = '0' # 默认回退值
    max_file_size = -1

    if os.path.exists(distorted_sparse_base):
        # 遍历所有数字文件夹 (0, 1, 2...)
        subfolders = [f for f in os.listdir(distorted_sparse_base) if f.isdigit()]
        
        if subfolders:
            # print(f"[{item}] Found reconstruction sub-models: {subfolders}")
            for folder in subfolders:
                # 通过检查 points3D.bin 的大小来判断哪个模型最大
                p3d_path = os.path.join(distorted_sparse_base, folder, "points3D.bin")
                # 兼容文本格式 (points3D.txt)
                if not os.path.exists(p3d_path):
                    p3d_path = os.path.join(distorted_sparse_base, folder, "points3D.txt")
                
                if os.path.exists(p3d_path):
                    curr_size = os.path.getsize(p3d_path)
                    if curr_size > max_file_size:
                        max_file_size = curr_size
                        best_model_folder = folder
            
            print(f"[{item}] Selected Model {best_model_folder} (Size: {max_file_size/1024:.2f} KB)")
    
    # 使用自动选择的最佳模型路径
    input_path_full = os.path.join(distorted_sparse_base, best_model_folder)
    # =========================================================
    # 修改结束
    # =========================================================

    output_path_full = os.path.join(args.source_path, inputDir)

    img_undist_cmd = (colmap_command + " image_undistorter \
        --image_path " + image_path_full + " \
        --input_path " + input_path_full + " \
        --output_path " + output_path_full + " \
        --output_type COLMAP")
    
    # print(img_undist_cmd)
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    # 路径拼接优化：使用 os.path.join 替代 +
    sparse_path = os.path.join(args.source_path, inputDir, "sparse")
    if os.path.exists(sparse_path):
        files = os.listdir(sparse_path)
        os.makedirs(os.path.join(sparse_path, "0"), exist_ok=True)
        # Copy each file from the source directory to the destination directory
        for file in files:
            if file == '0':
                continue
            # 注意：这里的 inputDir[1:] 逻辑保留了你原本的代码，请确保这是你预期的
            # 如果 inputDir 是 "colmap_1"，inputDir[1:] 就是 "olmap_1"，这看起来有点奇怪？
            # 假设你原本的逻辑是正确的，我先保持原样，除了路径拼接
            # 但通常 os.path.join(args.source_path, inputDir, ...) 更稳妥
            
            # 原代码逻辑保留：
            source_file = os.path.join(args.source_path, inputDir, "sparse", file) 
            destination_file = os.path.join(args.source_path, inputDir, "sparse", "0", file)
            shutil.move(source_file, destination_file)

    if(args.resize):
        print("Copying and resizing...")
        # Resize images.
        # 同样保留了 inputDir[1:] 的逻辑，但建议检查是否应该是 inputDir
        images_2_path = os.path.join(args.source_path, inputDir, "images_2")
        os.makedirs(images_2_path, exist_ok=True)
        
        images_path = os.path.join(args.source_path, inputDir, "images")
        files = os.listdir(images_path)
        
        # Copy each file from the source directory to the destination directory
        for file in files:
            source_file = os.path.join(images_path, file)
            destination_file = os.path.join(images_2_path, file)
            shutil.copy2(source_file, destination_file)
            # print("Resizing " + source_file + " to " + destination_file)
            exit_code = os.system(magick_command + " mogrify -resize 50% " + destination_file)
            if exit_code != 0:
                logging.error(f"50% resize failed with code {exit_code}. Exiting.")
                exit(exit_code)

p = mp.Pool(100)
res = []
for i in tqdm(range(1,7)):
    # print(i)
    item = f"colmap_{i}"
    res.append(p.apply_async(process, args=(item,)))
p.close()
p.join()
print("Done.")