import os
# ==========================================
# 补丁 1: 强制禁用 xformers
# 解决 "No operator found" 和版本不匹配导致的崩溃
import sys
sys.modules["xformers"] = None
sys.modules["xformers.ops"] = None
import logging
import argparse
from argparse import ArgumentParser
import shutil
import sqlite3
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import itertools

# 引入 RoMa
from romatch import roma_outdoor

# COLMAP 数据库工具类
class COLMAPDatabase:
    def __init__(self, path):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.cursor = self.conn.cursor()

    def get_image_ids_and_names(self):
        # 返回 {image_name: image_id}
        self.cursor.execute("SELECT image_id, name FROM images")
        return {name: image_id for image_id, name in self.cursor.fetchall()}

    def get_keypoints(self, image_id):
        # 获取特征点 (N, 2)
        self.cursor.execute("SELECT data, rows, cols FROM keypoints WHERE image_id=?", (image_id,))
        row = self.cursor.fetchone()
        if row is None:
            return np.zeros((0, 2), dtype=np.float32)
        data, rows, cols = row
        keypoints = np.frombuffer(data, dtype=np.float32).reshape(rows, cols)
        return keypoints[:, :2] # 只取 x, y

    def close(self):
        self.conn.close()

def run_roma_matching(db_path, image_path, output_match_file, use_gpu=True):
    print("Initializing RoMa for feature matching...")
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    
    # 初始化 RoMa 模型
    roma_model = roma_outdoor(device=device, coarse_res=560, upsample_res=(864, 1152))
    
    # ==========================================
    # 补丁 2: 强制禁用 custom local correlation
    # 解决 "No module named 'local_corr'" 错误
    # ==========================================
    print("Patching model to disable custom local_corr kernel...")
    if hasattr(roma_model, "decoder") and hasattr(roma_model.decoder, "conv_refiner"):
        refiners = roma_model.decoder.conv_refiner
        if isinstance(refiners, torch.nn.ModuleDict):
            for key, module in refiners.items():
                if hasattr(module, "use_custom_corr"):
                    module.use_custom_corr = False
        elif hasattr(refiners, "use_custom_corr"):
            refiners.use_custom_corr = False
    
    db = COLMAPDatabase(db_path)
    name_to_id = db.get_image_ids_and_names()
    image_names = list(name_to_id.keys())
    
    pairs = list(itertools.combinations(image_names, 2))
    print(f"Matching {len(pairs)} pairs with RoMa...")
    
    with open(output_match_file, 'w') as f:
        for name1, name2 in tqdm(pairs):
            id1 = name_to_id[name1]
            id2 = name_to_id[name2]
            
            kpts1 = db.get_keypoints(id1)
            kpts2 = db.get_keypoints(id2)
            
            if len(kpts1) == 0 or len(kpts2) == 0:
                continue

            im1_path = os.path.join(image_path, name1)
            im2_path = os.path.join(image_path, name2)
            
            try:
                with torch.inference_mode():
                    # 3. RoMa 匹配
                    warp, certainty = roma_model.match(im1_path, im2_path, device=device)
                    
                    # ==========================================
                    # 补丁 3: Unbatch & Split Symmetric Output
                    # 解决 "permute dim mismatch" 和 对称输出问题
                    # ==========================================
                    warp = warp[0]          
                    certainty = certainty[0] 
                    
                    W_roma = warp.shape[1] // 2
                    warp = warp[:, :W_roma, :]
                    certainty = certainty[:, :W_roma]
                    
                    H_A, W_A = np.array(Image.open(im1_path)).shape[:2]
                    H_B, W_B = np.array(Image.open(im2_path)).shape[:2]
                    
                    kpts1_norm = torch.from_numpy(kpts1.copy()).to(device) 
                    kpts1_norm = torch.stack(
                        (2 * kpts1_norm[:, 0] / W_A - 1, 2 * kpts1_norm[:, 1] / H_A - 1), dim=-1
                    )
                    
                    kpts2_norm = torch.from_numpy(kpts2.copy()).to(device)
                    kpts2_norm = torch.stack(
                        (2 * kpts2_norm[:, 0] / W_B - 1, 2 * kpts2_norm[:, 1] / H_B - 1), dim=-1
                    )
                    
                    # ==========================================
                    # 补丁 4: Fix Tuple Attribute Error
                    # 解决 "tuple object has no attribute cpu"
                    # ==========================================
                    inds = roma_model.match_keypoints(
                        kpts1_norm, kpts2_norm, warp, certainty, return_inds=True
                    )
                    
                    # 关键修复：match_keypoints 返回的是 tuple，需要 stack 起来
                    matches = torch.stack(inds, dim=1).cpu().numpy()
            
            except Exception as e:
                # print(f"Error matching {name1} and {name2}: {e}")
                continue

            if len(matches) > 15:
                f.write(f"{name1} {name2}\n")
                np.savetxt(f, matches, fmt='%d')
                f.write("\n")
    
    db.close()
    print(f"Matches saved to {output_match_file}")

if __name__ == "__main__":
    parser = ArgumentParser("Colmap converter with RoMa")
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

    if not args.skip_matching:
        distorted_sparse = os.path.join(args.source_path, "distorted", "sparse")
        os.makedirs(distorted_sparse, exist_ok=True)

        db_path = os.path.join(args.source_path, "input.db")
        input_path = os.path.join(args.source_path, "input")
        match_file = os.path.join(args.source_path, "roma_matches.txt")

        feat_extracton_cmd = (
            f'{colmap_command} feature_extractor '
            f'--database_path "{db_path}" '
            f'--image_path "{input_path}" '
            f'--ImageReader.single_camera 1 '
            f'--ImageReader.camera_model {args.camera} '
        )
        
        print(f"Running Feature Extraction: {feat_extracton_cmd}")
        exit_code = os.system(feat_extracton_cmd)
        if exit_code != 0:
            logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
            exit(exit_code)

        print("Running RoMa Matching...")
        run_roma_matching(db_path, input_path, match_file)
        
        match_import_cmd = (
            f'{colmap_command} matches_importer '
            f'--database_path "{db_path}" '
            f'--match_list_path "{match_file}" '
            f'--match_type raw '
        )
        
        print(f"Importing and verifying matches: {match_import_cmd}")
        exit_code = os.system(match_import_cmd)
        if exit_code != 0:
            logging.error(f"Matches importer failed with code {exit_code}. Exiting.")
            exit(exit_code)

        mapper_cmd = (
            f'{colmap_command} mapper '
            f'--database_path "{db_path}" '
            f'--image_path "{input_path}" '
            f'--output_path "{distorted_sparse}" '
            f'--Mapper.ba_global_function_tolerance=0.000001'
        )
        
        print(f"Running Mapper: {mapper_cmd}")
        exit_code = os.system(mapper_cmd)
        if exit_code != 0:
            logging.error(f"Mapper failed with code {exit_code}. Exiting.")
            exit(exit_code)

    # 5. Image undistortion
    # =========================================================
    # 补丁 5: 自动寻找最大的模型文件夹，而不是硬编码 "0"
    # =========================================================
    distorted_sparse_base = os.path.join(args.source_path, "distorted", "sparse")
    best_model_folder = None
    max_file_size = -1

    if os.path.exists(distorted_sparse_base):
        # 遍历所有数字文件夹 (0, 1...)
        subfolders = [f for f in os.listdir(distorted_sparse_base) if f.isdigit()]
        
        if not subfolders:
            print(f"Error: No reconstruction folders found in {distorted_sparse_base}")
        else:
            print(f"Found reconstruction sub-models: {subfolders}")
            for folder in subfolders:
                # 通过检查 points3D.bin 的大小来判断哪个模型最大
                p3d_path = os.path.join(distorted_sparse_base, folder, "points3D.bin")
                # 兼容文本格式
                if not os.path.exists(p3d_path):
                    p3d_path = os.path.join(distorted_sparse_base, folder, "points3D.txt")
                
                if os.path.exists(p3d_path):
                    curr_size = os.path.getsize(p3d_path)
                    print(f" - Model {folder}: size {curr_size/1024:.2f} KB")
                    if curr_size > max_file_size:
                        max_file_size = curr_size
                        best_model_folder = folder
            
            if best_model_folder is not None:
                print(f"==> Selecting Model {best_model_folder} as the main reconstruction.")
            else:
                best_model_folder = '0'
    
    # 设定最终读取的路径
    if best_model_folder:
        input_recon_path = os.path.join(distorted_sparse_base, best_model_folder)
    else:
        input_recon_path = os.path.join(distorted_sparse_base, "0")

    if os.path.exists(input_recon_path):
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
            
        sparse_dir = os.path.join(args.source_path, "sparse")
        sparse_0_dir = os.path.join(sparse_dir, "0")
        os.makedirs(sparse_0_dir, exist_ok=True)

        if os.path.exists(sparse_dir):
            files = os.listdir(sparse_dir)
            for file in files:
                if file == '0' or os.path.isdir(os.path.join(sparse_dir, file)):
                    continue
                source_file = os.path.join(sparse_dir, file)
                destination_file = os.path.join(sparse_0_dir, file)
                shutil.move(source_file, destination_file)
    else:
        print(f"Error: Reconstruction output not found at {input_recon_path}. Skipping undistortion.")

    if(args.resize):
        print("Copying and resizing...")
        # Resize images.
        os.makedirs(args.source_path + "/images_2", exist_ok=True)
        os.makedirs(args.source_path + "/images_4", exist_ok=True)
        os.makedirs(args.source_path + "/images_8", exist_ok=True)
        files = os.listdir(args.source_path + "/images")
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