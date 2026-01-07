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

# import os
# import logging
# from argparse import ArgumentParser
# import shutil

# # This Python script is based on the shell converter script provided in the MipNerF 360 repository.
# parser = ArgumentParser("Colmap converter")
# parser.add_argument("--no_gpu", action='store_true')
# parser.add_argument("--skip_matching", action='store_true')
# parser.add_argument("--source_path", "-s", required=True, type=str)
# parser.add_argument("--camera", default="OPENCV", type=str)
# parser.add_argument("--colmap_executable", default="", type=str)
# parser.add_argument("--resize", action="store_true")
# parser.add_argument("--magick_executable", default="", type=str)
# args = parser.parse_args()
# colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
# magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
# use_gpu = 1 if not args.no_gpu else 0

# if not args.skip_matching:
#     os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

#     ## Feature extraction
#     # feat_extracton_cmd = colmap_command + " feature_extractor "\
#     #     "--database_path " + args.source_path + "input.db \
#     #     --image_path " + args.source_path + "/input \
#     #     --ImageReader.single_camera 1 \
#     #     --ImageReader.camera_model " + args.camera + " \
#     #     --SiftExtraction.use_gpu " + str(use_gpu)
        
#     ## Feature extraction
#     db_path = os.path.join(args.source_path, "input.db")
#     input_path = os.path.join(args.source_path, "input")
    
#     # 注意：移除了 --SiftExtraction.use_gpu 选项以避免报错
#     feat_extracton_cmd = (
#         f'{colmap_command} feature_extractor '
#         f'--database_path "{db_path}" '
#         f'--image_path "{input_path}" '
#         f'--ImageReader.single_camera 1 '
#         f'--ImageReader.camera_model {args.camera} '
#     )
#     exit_code = os.system(feat_extracton_cmd)
    
    
#     if exit_code != 0:
#         logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
#         exit(exit_code)

#     ## Feature matching
#     # feat_matching_cmd = colmap_command + " exhaustive_matcher \
#     #     --database_path " + args.source_path + "input.db \
#     #     --SiftMatching.use_gpu " + str(use_gpu)
#     # 注意：移除了 --SiftMatching.use_gpu 选项
#     feat_matching_cmd = (
#         f'{colmap_command} exhaustive_matcher '
#         f'--database_path "{db_path}"'
#     )
#     exit_code = os.system(feat_matching_cmd)
#     if exit_code != 0:
#         logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
#         exit(exit_code)

#     ### Bundle adjustment
#     # The default Mapper tolerance is unnecessarily large,
#     # decreasing it speeds up bundle adjustment steps.
#     mapper_cmd = (colmap_command + " mapper \
#         --database_path " + args.source_path + "input.db \
#         --image_path "  + args.source_path + "/input \
#         --output_path "  + args.source_path + "/distorted/sparse \
#         --Mapper.ba_global_function_tolerance=0.000001")
#     exit_code = os.system(mapper_cmd)
#     if exit_code != 0:
#         logging.error(f"Mapper failed with code {exit_code}. Exiting.")
#         exit(exit_code)

# ### Image undistortion
# ## We need to undistort our images into ideal pinhole intrinsics.
# # img_undist_cmd = (colmap_command + " image_undistorter \
# #     --image_path " + args.source_path + "/input \
# #     --input_path " + args.source_path + "/distorted/sparse/0 \
# #     --output_path " + args.source_path + "\
# #     --output_type COLMAP")
# # 这里 input_path 指向 mapper 的输出 (distorted/sparse/0)

# input_recon_path = os.path.join(args.source_path, "distorted", "sparse", "0")
# img_undist_cmd = (
#     f'{colmap_command} image_undistorter '
#     f'--image_path "{os.path.join(args.source_path, "input")}" '
#     f'--input_path "{input_recon_path}" '
#     f'--output_path "{args.source_path}" '
#     f'--output_type COLMAP'
# )
# print(f"Running: {img_undist_cmd}")
# exit_code = os.system(img_undist_cmd)
# if exit_code != 0:
#     logging.error(f"Mapper failed with code {exit_code}. Exiting.")
#     exit(exit_code)
# # 整理文件结构：将生成的 sparse 模型移动到 sparse/0 下
# sparse_dir = os.path.join(args.source_path, "sparse")
# sparse_0_dir = os.path.join(sparse_dir, "0")
# os.makedirs(sparse_0_dir, exist_ok=True)

# files = os.listdir(args.source_path + "/sparse")
# # os.makedirs(args.source_path + "/sparse/0", exist_ok=True)

# # Copy each file from the source directory to the destination directory
# for file in files:
#     if file == '0':
#         continue
#     source_file = os.path.join(args.source_path, "sparse", file)
#     destination_file = os.path.join(args.source_path, "sparse", "0", file)
#     shutil.move(source_file, destination_file)
import os
import logging
from argparse import ArgumentParser
import shutil

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0

if not args.skip_matching:
    # 使用 os.path.join 确保路径正确
    distorted_sparse = os.path.join(args.source_path, "distorted", "sparse")
    os.makedirs(distorted_sparse, exist_ok=True)

    ## Feature extraction
    db_path = os.path.join(args.source_path, "input.db")
    input_path = os.path.join(args.source_path, "input")
    
    # 注意：移除了 --SiftExtraction.use_gpu 选项以避免报错
    feat_extracton_cmd = (
        f'{colmap_command} feature_extractor '
        f'--database_path "{db_path}" '
        f'--image_path "{input_path}" '
        f'--ImageReader.single_camera 1 '
        f'--ImageReader.camera_model {args.camera} '
    )
    
    print(f"Running: {feat_extracton_cmd}")
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching
    # 注意：移除了 --SiftMatching.use_gpu 选项
    feat_matching_cmd = (
        f'{colmap_command} exhaustive_matcher '
        f'--database_path "{db_path}"'
    )
    
    print(f"Running: {feat_matching_cmd}")
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Bundle adjustment
    mapper_cmd = (
        f'{colmap_command} mapper '
        f'--database_path "{db_path}" '
        f'--image_path "{input_path}" '
        f'--output_path "{distorted_sparse}" '
        f'--Mapper.ba_global_function_tolerance=0.000001'
    )
    
    print(f"Running: {mapper_cmd}")
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

### Image undistortion
## We need to undistort our images into ideal pinhole intrinsics.
# 这里 input_path 指向 mapper 的输出 (distorted/sparse/0)
input_recon_path = os.path.join(args.source_path, "distorted", "sparse", "0")
img_undist_cmd = (
    f'{colmap_command} image_undistorter '
    f'--image_path "{os.path.join(args.source_path, "input")}" '
    f'--input_path "{input_recon_path}" '
    f'--output_path "{args.source_path}" '
    f'--output_type COLMAP'
)

print(f"Running: {img_undist_cmd}")
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Image undistorter failed with code {exit_code}. Exiting.")
    exit(exit_code)

# 整理文件结构：将生成的 sparse 模型移动到 sparse/0 下
sparse_dir = os.path.join(args.source_path, "sparse")
sparse_0_dir = os.path.join(sparse_dir, "0")
os.makedirs(sparse_0_dir, exist_ok=True)

files = os.listdir(sparse_dir)
for file in files:
    if file == '0':
        continue
    source_file = os.path.join(sparse_dir, file)
    destination_file = os.path.join(sparse_0_dir, file)
    shutil.move(source_file, destination_file)

if(args.resize):
    print("Copying and resizing...")

    # Resize images.
    os.makedirs(args.source_path + "/images_2", exist_ok=True)
    os.makedirs(args.source_path + "/images_4", exist_ok=True)
    os.makedirs(args.source_path + "/images_8", exist_ok=True)
    # Get the list of files in the source directory
    files = os.listdir(args.source_path + "/images")
    # Copy each file from the source directory to the destination directory
    for file in files:
        source_file = os.path.join(args.source_path, "images", file)

        destination_file = os.path.join(args.source_path, "images_2", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 50% " + destination_file)
        if exit_code != 0:
            logging.error(f"50% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_4", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 25% " + destination_file)
        if exit_code != 0:
            logging.error(f"25% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_8", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 12.5% " + destination_file)
        if exit_code != 0:
            logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

print("Done.hahaha")
