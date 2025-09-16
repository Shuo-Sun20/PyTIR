"""
记录每个仓库的LaRY相关信息, 包括ClusterNum, EntityNodeNum, AnalyzeTime, 保存到CSV文件中
"""
import json
import csv
from type_llm.utils.config import EntityGraph_Path
from type_llm.methods.full_LARRY.Entity_Graph import Entity_Graph

def dict_to_csv(data, filename='output.csv'):
    """
    将嵌套字典转换为CSV文件
    
    参数:
    data: 嵌套字典数据
    filename: 输出的CSV文件名
    """
    # 提取所有可能的字段名（排除第一个键名字段）
    fieldnames = set()
    for item in data.values():
        fieldnames.update(item.keys())
    
    # 将字段名排序，确保一致的顺序
    fieldnames = sorted(fieldnames)
    
    # 写入CSV文件
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['name'] + fieldnames)
        
        # 写入表头
        writer.writeheader()
        
        # 写入数据行
        for name, metrics in data.items():
            row = {'name': name}
            row.update(metrics)
            writer.writerow(row)


if __name__ == '__main__':
    data = json.load(open('/mnt/data2/Users/sunshuo/type_LLM_V_7.11/data/results/csv_data.json'))
    dict_data = {
        project_name:{
            "AnalyzeTime": seq[-1][0],
            "ClusterNum": seq[-1][1],
            "Added_Edge": seq[-1][2],
            "depTime": seq[-1][3],
            "infTime": seq[-1][4],
            "repairTime": seq[-1][5],
            "lastTryTime": seq[-1][6],
        }
        for project_name, seq in data.items()    }
    for project in dict_data:
        full_path = EntityGraph_Path / f"{project}.json"
        with open(full_path,'r') as f:
                data = json.load(f)
        graph = Entity_Graph(**data) 
        # graph.preprocess()
        edgeNum = 0
        for node in graph.node_dict.values():
            edgeNum += len(node.dependency)
        dict_data[project]['EdgeNum'] = edgeNum + dict_data[project]["Added_Edge"]
        dict_data[project]['EntityNodNum'] = len(graph.node_dict)
        dict_data[project]['AvgClusterSize'] = dict_data[project]['EntityNodNum'] / dict_data[project]['ClusterNum']
    dict_to_csv(dict_data, "/mnt/data2/Users/sunshuo/type_LLM_V_7.11/data/results/LaRY_Repo_Info.csv")