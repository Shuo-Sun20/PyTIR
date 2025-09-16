import os
import shutil
from collections import defaultdict

src = '/mnt/data2/Users/sunshuo/type_LLM_V_7.11/data/intermediate/validation_NoCheck'     # 源目录路径
dest = '/mnt/data2/Users/sunshuo/type_LLM_V_7.11/data/result_repo/LaRY_NoCheck'   # 目标目录路径

# 确保目标目录存在
os.makedirs(dest, exist_ok=True)

# 存储每个project的最大num和对应的文件夹路径
project_dict = defaultdict(lambda: {'max_num': -1, 'folder': None})

# 遍历src目录中的所有条目
for entry in os.listdir(src):
    folder_path = os.path.join(src, entry)
    
    # 跳过非目录文件
    if not os.path.isdir(folder_path):
        continue
    
    # 拆分文件夹名（处理多个下划线的情况）
    parts = entry.split('_')
    if len(parts) < 2:
        continue
    
    # 获取项目名和数字部分
    num_part = parts[-1]
    project_name = '_'.join(parts[:-1])
    
    # 验证数字部分
    if not num_part.isdigit():
        continue
    
    num = int(num_part)
    
    # 更新当前项目的最大num记录
    if num > project_dict[project_name]['max_num']:
        project_dict[project_name]['max_num'] = num
        project_dict[project_name]['folder'] = entry

# 复制文件夹到目标目录
for project, data in project_dict.items():
    if data['folder']:
        src_path = os.path.join(src, data['folder'])
        dest_path = os.path.join(dest, project, project )
        
        # 删除目标文件夹（如果存在）
        if os.path.exists(dest_path):
            shutil.rmtree(dest_path)
        
        # 复制整个文件夹
        shutil.copytree(src_path, dest_path)
        print(f"Copied: {project} -> {dest_path}")
    else:
        print(f'{project} not found')