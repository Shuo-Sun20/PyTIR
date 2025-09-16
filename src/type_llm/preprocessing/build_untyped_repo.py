import os
import shutil
import ast
import re
from typing import Union
from type_llm.utils.config import originalProjectsDir, untypedProjectsDir, projects
from type_llm.utils import funcTransformer

def x_copy(srcDir, dstDir):
    if os.path.exists(dstDir):
        shutil.rmtree(dstDir)
    shutil.copytree(srcDir, dstDir)


class TypeAnnotationRemover(funcTransformer):
    def func_trans(self, node):
        #移除overload函数
        is_overload = False
        for decorator in node.decorator_list:
            if (isinstance(decorator, ast.Name) and decorator.id == "overload") \
                or (isinstance(decorator, ast.Attribute) and decorator.attr == "overload"):
                is_overload = True
                break
        if is_overload:
            return None
        
        # 移除函数参数和返回类型注释
        node = self._process_function_args(node)
        node.returns = None
        
        # 处理函数体中的类型注释和docstring
        node.body = [self.visit(stmt) for stmt in node.body]
        self._process_docstring(node.body)
        
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Union[ast.Assign, None]:
        # 转换类型注释赋值为普通赋值（保留赋值值）
        if node.value is not None:
            return ast.Assign(targets=[node.target], value=node.value)
        else:
            return node

    def _process_function_args(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]):
        # 处理所有类型的参数注释
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
        return node

    def _process_docstring(self, body: list):
        # 处理docstring节点
        if not body:
            return
        
        first_node = body[0]
        if isinstance(first_node, ast.Expr) and isinstance(first_node.value, ast.Constant):
            docstring = first_node.value.value
            if isinstance(docstring, str):
                cleaned = self._clean_docstring(docstring)
                first_node.value.value = cleaned

    def _clean_docstring(self, docstring: str) -> str:
        # 使用正则表达式清理docstring中的类型信息
        patterns = [
            (r':param\s+\w+\s+(\w+):', r':param \1:'),    # 移除参数类型
            (r':type\s+\w+:.*', ''),                      # 移除类型声明行
            (r':rtype:.*(\n|$)', ''),                     # 移除返回类型
            (r'(\w+)\s*$[^)]*$(.*)', r'\1\2'),            # 移除括号内的类型说明
            (r'\s*->\s.*?:', ':')                         # 移除返回箭头注释
        ]
        for pattern, replacement in patterns:
            docstring = re.sub(pattern, replacement, docstring)
        return docstring.strip()

def process_file(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()
    try:
        tree = ast.parse(source)
        transformer = TypeAnnotationRemover()
        modified_tree = transformer.visit(tree)
        ast.fix_missing_locations(modified_tree)
        
        # 使用Python 3.9+内置的unparse
        new_code = ast.unparse(modified_tree)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_code)
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

def process_directory(directory: str):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                process_file(os.path.join(root, file))
                
if __name__ == "__main__":
    for project in projects:
        x_copy(originalProjectsDir / project, untypedProjectsDir / project)
        process_directory(untypedProjectsDir / project)