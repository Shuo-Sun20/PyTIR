from pathlib import Path


# projects = ['pre_commit_hooks','werkzeug','flake8','flask','click','fastapi','urllib3','black','pre_commit','typer','rich','jinja2']
projects = ['pre_commit_hooks']

#Config your LLM Information Here
BASE_URL  = 'XXXX'
MODEL = 'XXXX'
API_KEY = [
    'XXXX' ,
]


#rootPath的设置依赖于项目架构，config.py的路径改变时要改变rootPath
rootPath = Path(__file__).parent.parent.parent.parent
dataPath = rootPath / "data"
ThirdPartyPath = rootPath / "third_party"
illustrate_path = rootPath / "illustration"

#结果和日志的存储路径
resultsPath = dataPath / "results"
logPath = dataPath / "logs"

#中间结果的存储路径
intermediatePath = dataPath / "intermediate"
Raw_PyAnalyzer = intermediatePath / "Raw_PyAnalyzer"
EntityGraph_Path= intermediatePath / "EntityGraph"
Validation_Path = intermediatePath /  "validation"
Scope_Path = intermediatePath / "scope"
LLM_Result_Path = intermediatePath / "LLM_Results"
bak_Path = dataPath / 'intermediate_bak' / "validation"
#benchmark的存储路径
projectsDir = dataPath / "projects"
originalProjectsDir = projectsDir / "original"
untypedProjectsDir = projectsDir / "untyped"
prototype_Path = projectsDir / "untyped"
#evaluation
evaluationDir = dataPath / "evaluation"
evalProjectsDir = evaluationDir / "projects"
evalResultsDir = evaluationDir / "results"
evalOutputDir = evaluationDir / "EvalResults"