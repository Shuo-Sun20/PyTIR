from type_llm.methods.full_LARRY.Validation import run_mypy_strict
from type_llm.utils.log_manager import setup_logger
from type_llm.utils.config import evalResultsDir, evalOutputDir
from pathlib import Path
import os
import shutil
import tempfile
import json

logger = setup_logger(with_console=True)
detailed_errors = {}
all_res = "method,project,errors\n"
root_dir = evalResultsDir
for method in os.listdir(root_dir):
    method_dir = root_dir / method
    detailed_errors[method] = {}
    for project in os.listdir(method_dir):
        logger.info(f"Executing {method}:{project}")
        project_dir = method_dir / project 
        with tempfile.TemporaryDirectory() as temp_dir:
            target_dir = Path(temp_dir) / project
            shutil.copytree(project_dir, target_dir )
            errors = run_mypy_strict(target_dir)
            one_res = f"{method},{project},{len(errors)}\n"
            all_res += one_res
            logger.info(one_res)
            detailed_errors[method][project] = errors

if not evalOutputDir.exists():
    evalOutputDir.mkdir(parents=True)
with open(evalOutputDir / "errors.csv","w") as f:
    f.write(all_res)
with open(evalOutputDir /"detailed_errors.json","w") as f:
    json.dump(detailed_errors, f, indent=4)