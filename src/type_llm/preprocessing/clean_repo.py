import os
from pathlib import Path
from type_llm.utils.config import originalProjectsDir

def is_test_file(file_path):
    """识别测试文件（test_开头或_test结尾的.py文件）"""
    stem = file_path.stem
    return stem.startswith('test_') or stem.endswith('_test')

def is_irrelevant_dir(dir_path):
    """识别无关目录（docs、tests、__pycache__等）"""
    return 'test' in str(dir_path) \
        or 'docs' in str(dir_path) \
        or 'cookbook' in str(dir_path) \
        or 'example' in str(dir_path)
        
        
def clean_project(directory):
    """
    清理项目目录：删除非.py文件、测试文件和空目录
    
    参数：
        directory (str/Path): 要清理的目标目录路径
    
    异常：
        FileNotFoundError: 目录不存在时抛出
        ValueError: 路径不是目录时抛出
    """
    target_dir = Path(directory)
    
    # 验证路径有效性
    if not target_dir.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")
    if not target_dir.is_dir():
        raise ValueError(f"路径不是目录: {directory}")

    # 删除符合条件的文件
    for file_path in target_dir.rglob('*'):
        if file_path.is_file():
            # 判断是否非.py文件或测试文件
            if (file_path.suffix != '.py') or (file_path.suffix == '.py' and is_test_file(file_path)):
                try:
                    file_path.unlink()
                    print(f"🗑️ 已删除文件: {file_path}")
                except Exception as e:
                    print(f"❌ 删除失败 [{file_path}]: {str(e)}")

    # 删除空目录（从最深层开始）
    dir_list = []
    for dir_path in target_dir.rglob('*'):
        if dir_path.is_dir() and dir_path != target_dir:
            dir_list.append(dir_path)
    
    # 按目录深度倒序排序
    for dir_path in sorted(dir_list, key=lambda x: len(x.parts), reverse=True):
        if len(os.listdir(dir_path)) == 0:
            dir_path.rmdir()
            print(f"已清理空目录: {dir_path}")

if __name__ == "__main__":
    clean_project(originalProjectsDir)
