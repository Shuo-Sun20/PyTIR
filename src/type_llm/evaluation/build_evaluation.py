from type_llm.utils.config import originalProjectsDir,untypedProjectsDir, projects, evalProjectsDir, evalResultsDir, Validation_Path
import shutil


if __name__ == '__main__':
        resultDir = evalResultsDir / 'PyTIR'
        for project in projects:
            projectDir = evalProjectsDir / project
            shutil.copytree(originalProjectsDir / project, projectDir / 'original' , dirs_exist_ok=True)
            shutil.copytree(untypedProjectsDir / project, projectDir / 'untyped', dirs_exist_ok=True)
            shutil.copytree(Validation_Path / project, resultDir / project, dirs_exist_ok=True)