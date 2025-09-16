import pandas as pd
from pathlib import Path
from type_llm.utils.config import evalResultsDir as src_dir, evalOutputDir as output_dir
# 配置路径
output_dir.mkdir(exist_ok=True)  # 确保输出目录存在

# 遍历每个method文件夹
for method_dir in src_dir.iterdir():
    if not method_dir.is_dir():
        continue  # 跳过非目录文件
    
    method_name = method_dir.name
    all_dfs = []  # 存储当前method下所有CSV的DataFrame
    
    # 遍历当前method下的project文件夹
    for project_dir in method_dir.iterdir():
        if not project_dir.is_dir():
            continue
        
        # 查找project目录中的CSV文件（取第一个找到的CSV）
        csv_files = list(project_dir.glob("*.csv"))
        if not csv_files:
            continue
            
        csv_path = csv_files[0]  # 获取第一个CSV文件
        try:
            # 读取CSV并添加到列表（不自动添加索引）
            df = pd.read_csv(csv_path)
            # 添加method和project列标识来源
            df.insert(0, "project", project_dir.name)
            df.insert(0, "method", method_name)
            all_dfs.append(df)
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
    
    # 合并当前method的所有DataFrame
    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        output_path = output_dir / f"{method_name}_acc.csv"
        merged_df.to_csv(output_path, index=False)
        print(f"Saved merged data for [{method_name}] -> {output_path}")
    else:
        print(f"No CSV files found for method: {method_name}")

print("All methods processed.")