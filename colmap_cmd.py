import os
import subprocess
import sys
from pathlib import Path

# ==================== 配置参数区域 ====================
# 在这里直接设置参数，无需命令行传参
DATASET_PATH = "./"  # 数据集根目录 (必须包含 images 子文件夹)
OUTPUT_TYPE = "TXT"  # 输出格式: 'TXT' (文本) 或 'BIN' (二进制)
COLMAP_EXE = "colmap"  # COLMAP 可执行命令路径
# =====================================================

def run_command(cmd, step_name):
    """
    执行系统命令并处理输出
    """
    print(f"\n{'='*20} 正在执行步骤: {step_name} {'='*20}")
    print(f"指令: {cmd}")
    
    try:
        # 使用 shell=True 允许直接运行字符串命令
        # check=True 会在命令返回非零退出码时抛出异常
        subprocess.run(cmd, shell=True, check=True)
        print(f"✅ {step_name} 完成。")
    except subprocess.CalledProcessError as e:
        print(f"❌ {step_name} 失败！错误代码: {e.returncode}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")
        sys.exit(1)

def main():
    # 1. 路径配置（使用顶部定义的变量）
    root_path = Path(DATASET_PATH).resolve()
    images_path = root_path / "images"
    database_path = root_path / "database.db"
    sparse_path = root_path / "sparse"
    
    # 检查输入
    if not images_path.exists():
        print(f"❌ 错误: 在路径 {root_path} 下找不到 'images' 文件夹。")
        sys.exit(1)

    # 确保输出目录存在
    os.makedirs(sparse_path, exist_ok=True)

    # ---------------------------------------------------------
    # 2. 特征提取 (Feature Extraction)
    # ---------------------------------------------------------
    cmd_extract = (
        f"{COLMAP_EXE} feature_extractor "
        f"--database_path \"{database_path}\" "
        f"--image_path \"{images_path}\""
    )
    run_command(cmd_extract, "1. 特征提取")

    # ---------------------------------------------------------
    # 3. 特征匹配 (Feature Matching)
    # ---------------------------------------------------------
    # 注意：exhaustive_matcher 适合图片较少的情况。如果图片非常多，建议改为 vocab_tree_matcher
    cmd_match = (
        f"{COLMAP_EXE} exhaustive_matcher "
        f"--database_path \"{database_path}\""
    )
    run_command(cmd_match, "2. 特征匹配")

    # ---------------------------------------------------------
    # 4. 稀疏重建 (Sparse Reconstruction / Mapper)
    # ---------------------------------------------------------
    """
    cmd_mapper = (
        f"{COLMAP_EXE} mapper "
        f"--database_path \"{database_path}\" "
        f"--image_path \"{images_path}\" "
        f"--output_path \"{sparse_path}\""
    )
    """
    # 测试使用：添加了 --Mapper.init_min_num_inliers 10 等参数来降低初始化门槛
    cmd_mapper = (
        f"{COLMAP_EXE} mapper "
        f"--database_path \"{database_path}\" "
        f"--image_path \"{images_path}\" "
        f"--output_path \"{sparse_path}\" "
        f"--Mapper.init_min_num_inliers 10 "
        f"--Mapper.init_max_error 8.0 "
        f"--Mapper.init_max_forward_motion 0.95 "
        f"--Mapper.init_min_tri_angle 4.0"
    )
    run_command(cmd_mapper, "3. 稀疏重建")

    # ---------------------------------------------------------
    # 5. 格式转换 (Model Converter)
    # ---------------------------------------------------------
    # Mapper 默认会在 sparse 下创建一个名为 '0' 的文件夹
    input_model_path = sparse_path / "0"
    
    if not input_model_path.exists():
        print("❌ 错误: 稀疏重建未能生成模型文件夹 '0'。可能是图像无法匹配。")
        sys.exit(1)

    # 根据用户选择决定输出文件夹名称
    if OUTPUT_TYPE == "TXT":
        output_model_path = sparse_path / "0_text"
    else:
        output_model_path = sparse_path / "0_bin_converted"

    os.makedirs(output_model_path, exist_ok=True)

    cmd_convert = (
        f"{COLMAP_EXE} model_converter "
        f"--input_path \"{input_model_path}\" "
        f"--output_path \"{output_model_path}\" "
        f"--output_type {OUTPUT_TYPE}"
    )
    run_command(cmd_convert, f"4. 模型格式转换 -> {OUTPUT_TYPE}")

    print(f"\n🎉 全部流程完成！")
    print(f"📂 最终模型保存在: {output_model_path}")
    if OUTPUT_TYPE == "TXT":
        print(f"   包含文件: cameras.txt, images.txt, points3D.txt")
    else:
        print(f"   包含文件: cameras.bin, images.bin, points3D.bin")

if __name__ == "__main__":
    main()
# 使用方法：
# 1. 在文件顶部的配置区域修改参数
# 2. 直接运行: python colmap_cmd.py