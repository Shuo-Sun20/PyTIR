import subprocess
from type_llm.utils.log_manager import setup_logger
import os
from type_llm.utils import silent_Write,silent_JDump
import json
from type_llm.utils.config import prototype_Path, logPath, Raw_PyAnalyzer, projects, ThirdPartyPath
logger = setup_logger()



def pyanalyzer_execute(project, project_dir=prototype_Path, log_dir=logPath, ele_data_dir=Raw_PyAnalyzer):
    file_path = project_dir / project
    
    # 进入 third_party 目录
    os.chdir(ThirdPartyPath)
    
    # 执行 PyAnalyzer 命令
    logger.info(f"Running PyAnalyzer on {file_path}")
    result = subprocess.run(
        ["python", '-m', 'pyanalyzer', file_path],
        capture_output=True,
        text=True,
    )
    
    #save log
    logger.info(f"Saving Results")
    logFile = log_dir / 'eleSplit_log' / f'{project}.log'
    silent_Write(result.stdout, logFile)

    #save ele data
    output_file = f'{project}-report-PyAnalyzer.json'
    if not os.path.exists(output_file):
        logger.error(f'output file {output_file} not found')
    else:
        dest_file = ele_data_dir / f'{project}.json'
        with open(output_file, 'r') as f:
            ele_data = json.load(f)
        silent_JDump(ele_data, dest_file)      
      
    logger.info(f"PyAnalyzer Done on {file_path}")

if __name__ == '__main__':
    for project in projects:
        pyanalyzer_execute(project)


