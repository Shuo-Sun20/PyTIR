
from ..utils.config import projects, repoMetaInfo_dir, repoDetailedInfo_dir, package_path
from ..utils import setup_logger
from ..utils.validation import run_mypy,extract_comment,restore_comments
from pathlib import Path
import shutil
import ast
import csv
import json
import os
logger = setup_logger()

def extract_testSet(project):
    class TestSet_Extractor(ast.NodeVisitor):
        def __init__(self, fileName):
            self.current_name = "global@global"
            self.file_name = fileName
            self.all_snippet = []
            self.black_list = set()
            
        def visit_AnnAssign(self, node):
            self.all_snippet.append(
                {
                    "cat": "builtins",
                    'file': str(self.file_name),
                    "generic": False,
                    "gttype": "str",
                    "loc":  self.current_name,
                    "name": ast.unparse(node.target),
                    "origttype": "builtins.str",
                    "processed_gttype": "str",
                    "scope": "local",
                    "type_depth": 0
                }
            )
            
        def extract_arg(self, arg):
            if arg.annotation:
                    self.all_snippet.append(
                        {
                            "cat": "builtins",
                            'file': str(self.file_name),
                            "generic": False,
                            "gttype": "str",
                            "loc":  self.current_name,
                            "name": arg.arg,
                            "origttype": "builtins.str",
                            "processed_gttype": "str",
                            "scope": "arg",
                            "type_depth": 0
                        }
                    )
        def visit_FunctionDef(self, node):
            # 移除函数参数和返回类型注释
            if node.name == 'main':
                pass
            if ("overload" in ast.unparse(node.decorator_list)) or ('property' in ast.unparse(node.decorator_list)) or ('setter' in ast.unparse(node.decorator_list)):
                self.black_list.add(node.name+'@'+self.current_name.split('@')[1])
                return
            old_stack = self.current_name
            if node.name+'@'+old_stack.split('@')[1] in self.black_list:
                return
            self.current_name = node.name+'@'+old_stack.split('@')[1]
            for arg in node.args.posonlyargs:
                self.extract_arg(arg)
            for arg in node.args.args:
                self.extract_arg(arg)
            if node.args.vararg:
                self.extract_arg(node.args.vararg)
            for arg in node.args.kwonlyargs:
                self.extract_arg(arg)
            if node.args.kwarg:
                self.extract_arg(node.args.kwarg)
            if node.returns:
                self.all_snippet.append(
                        {
                            "cat": "builtins",
                            'file': str(self.file_name),
                            "generic": False,
                            "gttype": "str",
                            "loc":  self.current_name,
                            "name": node.name,
                            "origttype": "builtins.str",
                            "processed_gttype": "str",
                            "scope": "return",
                            "type_depth": 0
                        }
                    )     
            self.current_name = old_stack
        
        def visit_ClassDef(self, node):
            old_stack = self.current_name
            self.current_name = 'global@'+node.name
            super().generic_visit(node)
            self.current_name = old_stack
        
        def visit_AsyncFunctionDef(self, node):
            self.visit_FunctionDef(node)
    
    all_snippet = []
    source_dir = Path(f"/mnt/data2/Users/sunshuo/type_LLM_V_7.11/data/benchmarks/benchmark_repositories/{project}/original/{project}")
    for py_file in source_dir.rglob("*.py"):
        te = TestSet_Extractor(py_file)
        node = ast.parse(open(py_file).read())
        te.visit(node)
        all_snippet.extend(te.all_snippet)
    
    target_dir = Path(f"/mnt/data2/Users/sunshuo/type_LLM_V_7.11/data/DLDataSet/{project}")
    target_dir.mkdir(exist_ok=True)
    json.dump(all_snippet, open(target_dir/"testset.json",'w'),indent=4)
if __name__ == "__main__":
    for project in projects:
        extract_testSet(project)
        