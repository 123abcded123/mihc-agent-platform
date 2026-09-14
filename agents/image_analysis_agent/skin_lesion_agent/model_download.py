"""
皮肤病灶分割模型权重下载模块。

职责：
- 使用 gdown 从 Google Drive 下载 U-Net 权重文件；
- 若权重文件已存在则跳过下载，避免重复网络请求；
- 自动创建目标目录，供皮肤病灶分割推理模块调用。
"""
import os
import gdown

def download_model_checkpoint(gdrive_file_id, output_path):
    """
    Download model checkpoint from Google Drive if it doesn't exist.
    
    Args:
        gdrive_file_id (str): Google Drive file ID
        output_path (str): Path where model will be saved
    """
    # 若权重文件不存在，则从 Google Drive 下载；已存在则跳过。
    # 参数:
    #     gdrive_file_id: Google Drive 文件 ID
    #     output_path: 权重文件保存路径
    # Ensure the directory exists
    # 确保目标目录存在（不存在则递归创建）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Check if file already exists
    # 若权重文件已存在则直接返回，避免重复下载
    if not os.path.exists(output_path):
        print(f"Downloading model checkpoint to {output_path}...")
        # 拼接 Google Drive 直链并调用 gdown 下载权重文件
        url = f'https://drive.google.com/uc?id={gdrive_file_id}'
        gdown.download(url, output_path, quiet=False)
        print("Download complete!")

# Usage in app.py
# download_model_checkpoint('your_gdrive_file_id', 'path/to/checkpoint.pth')
