
from ..utils.config import benchRepo_dir, repoMetaInfo_dir, repoDetailedInfo_dir, package_path
from ..utils import setup_logger
from ..utils.validation import run_mypy,extract_comment,restore_comments
from pathlib import Path
import shutil
import ast
import csv
import json

logger = setup_logger()

def copy_py_files(src: Path, dst: Path, post_processor, keep_comments=True):
    # 创建目标目录（如果不存在）
    dst.mkdir(parents=True, exist_ok=True)
    
    # 遍历所有源目录中的.py文件
    for py_file in src.rglob("*.py"):
        # 获取相对于源目录的相对路径
        relative_path = py_file.relative_to(src)
        
        # 构建目标路径
        dest_file = dst / relative_path
        
        # 确保目标目录存在
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 复制文件
        with open(py_file, 'r',encoding='utf-8', errors='ignore') as src_file:
            with open(dest_file, 'w') as dst_file:
                source_code = src_file.read()
                if keep_comments:
                    comment_dict = extract_comment(source_code)
                    astNode = ast.parse(source_code)
                    content, offset = post_processor(astNode)
                    content = restore_comments(content, comment_dict, offset)
                else:
                    astNode = ast.parse(source_code)
                    content, _ = post_processor(astNode)
                dst_file.write(content)

def fix_mypy_builtin_errors(ast_node):
    content = ""
    ori_content = ast.unparse(ast_node)
    for line in ori_content.splitlines(True):
        if ('ctypes' in line and 'import' in line):
            content += line.strip('\n')+f'#type: ignore\n'
        else:
            content += line
    return content, []


def fix_error(file_name, line_number, error_code):
    content = ""
    with open(file_name) as f:
        for i, line in enumerate(f.readlines()):
            if (i == line_number - 1):
                content += line.strip('\n')+f'#type: ignore\n'
            else:
                content += line
    return content


def patch_mypy_errors(repo_dir):
    mypy_errors = run_mypy(repo_dir)
    for errorInfo in mypy_errors:
        error_file = errorInfo['file_path']
        error_line = errorInfo['line_number']
        error_code = errorInfo['error_code']
        fixed_content = fix_error(error_file, error_line, error_code)
        with open(error_file, 'w') as f:
            f.write(fixed_content)
    new_errors = run_mypy(repo_dir)
    if new_errors:
        logger.error(f"{len(new_errors)} unfixed errors in repo {repo}")
    return mypy_errors

def remove_node_ann(node):
    class Ann_Remover(ast.NodeTransformer):
        
        def visit_AnnAssign(self, node):
            if node.value:
                return ast.Assign(targets=[node.target], value=node.value)
            else:
                node.annotation = ast.Name(id='Unknown_Type',ctx=ast.Load())
                return node
            
        def visit_FunctionDef(self, node):
            # 移除函数参数和返回类型注释
            for arg in node.args.posonlyargs:
                arg.annotation = None
            for arg in node.args.args:
                arg.annotation = None
            if node.args.vararg:
                node.args.vararg.annotation = None
            for arg in node.args.kwonlyargs:
                arg.annotation = None
            if node.args.kwarg:
                node.args.kwarg.annotation = None
            node.returns = None
            
            return node
        
        def visit_AsyncFunctionDef(self, node):
            return self.visit_FunctionDef(node)
    
    modified_node = Ann_Remover().visit(node)
    importAnyStmt  = ast.parse("from typing import Any as Unknown_Type").body[0]
    modified_node.body.insert(2, importAnyStmt)
    modified_node = ast.fix_missing_locations(modified_node)
    return ast.unparse(modified_node), [(2,1)]

def create_original_repo(repo):
    """构建包含注释的groundtruth仓库，要求mypy不能报错。
    :param repo: 仓库名称：str
    """
    target_dir = package_path / repo
    dest_dir = benchRepo_dir / repo / 'original' / repo
    if not target_dir.exists():
        logger.warning(f"Repository {repo} does not exist in the package location.")
        return
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    copy_py_files(target_dir, dest_dir,fix_mypy_builtin_errors, False)
    return patch_mypy_errors(dest_dir)

def create_untyped_repo(repo):        
    """构建不含注释的input仓库，要求mypy不能报错。
    :param repo: 仓库名称：str
    """
    original_benchmark_dir = benchRepo_dir / repo / 'original' 
    untyped_benchmark_dir = benchRepo_dir / repo / 'untyped'
    if not original_benchmark_dir.exists():
        logger.warning(f"Repository {repo} does not exist in the Original Dir.")
        return []
    if untyped_benchmark_dir.exists():
        shutil.rmtree(untyped_benchmark_dir)
    copy_py_files(original_benchmark_dir, untyped_benchmark_dir, remove_node_ann, True)
    return patch_mypy_errors(untyped_benchmark_dir)

if __name__ == "__main__":
    candidate_repository_list_file = repoMetaInfo_dir / 'repo_list.txt'
    with open(candidate_repository_list_file, 'r') as f:
        repo_list = f.readlines()
        repo_list = [repo.strip() for repo in repo_list]
    summary_info = [["project", "errors in original repo", "errors in untyped repo"]]
    for repo in repo_list:
        logger.info(f"Processing repository: {repo}")
        original_errors = create_original_repo(repo)
        logger.info(f"Finished create original repository: {repo}")
        untyped_errors = create_untyped_repo(repo)
        logger.info(f"Finished create untyped repository: {repo}")
        Detailed_info = {"errors in original repo":original_errors, "error in untyped repo":untyped_errors}
        summary_info.append([repo, len(original_errors), len(untyped_errors)])
        with open(repoDetailedInfo_dir / f'{repo}.json', 'w') as f:
            json.dump(Detailed_info, f, indent=4)
    with open(repoMetaInfo_dir / 'summary.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(summary_info)
        