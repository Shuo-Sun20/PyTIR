import ast
from pathlib import Path
from type_llm.utils.config import untypedProjectsDir, prototype_Path, projects
import shutil
import os
import subprocess
import re
from type_llm.utils.log_manager import setup_logger

logger = setup_logger()

def extract_names(node):
    if isinstance(node, ast.Name):
        return [node.id]
    elif isinstance(node, ast.Tuple):
        names = []
        for name_node in node.elts:
            name = extract_names(name_node)
            names.extend(name)
        return names
    return []

class TypeAnnotationTransformer(ast.NodeTransformer):
    def __init__(self):
        self.in_global_scope = True
        self.all_objs = set()
        self.in_class_scope = False
        
    def visit(self, node):
        if '__all__' in ast.unparse(node):
            pass
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if self.in_global_scope:
                self.all_objs.add(node.name)
            old_scope = self.in_global_scope
            self.in_global_scope = False
            RET  = self._process_function(node)
            self.in_global_scope = old_scope
            # return RET
            return node
        
        elif isinstance(node, ast.ClassDef):
            if self.in_global_scope:
                self.all_objs.add(node.name)
            old_scope = self.in_global_scope
            self.in_global_scope = False
            RET = self._process_class(node)
            self.in_global_scope = old_scope
            # return RET
            return node
        
        elif isinstance(node, (ast.Import,ast.ImportFrom)):
            for alias in node.names:
                if not alias.asname and '.' not in alias.name:
                   alias.asname = alias.name
            if self.in_global_scope:
                for alias in node.names:
                    if alias.asname:
                        self.all_objs.add(alias.asname)
                    elif '.' not in alias.name:
                        self.all_objs.add(alias.name)
            return node
        
        elif isinstance(node, ast.Assign):
            all_names = []
            if isinstance(node.targets[0], ast.Name) and node.targets[0].id == '__all__':
                return None
            for target in node.targets:
                all_names.extend(extract_names(target))
            if self.in_global_scope:
                self.all_objs.update(all_names)
                
            keep_value = True
            for target in node.targets:
                if isinstance(target, ast.Tuple):
                    keep_value = False
            return node
            if keep_value: 
                return [ast.AnnAssign(target=ast.Name(id=name, ctx=ast.Store()),
                                    annotation=ast.Name(id='Any', ctx=ast.Load()), 
                                    value = node.value,
                                    simple=1) for name in all_names]
            else:
                return [ast.AnnAssign(target=ast.Name(id=name, ctx=ast.Store()),
                                    annotation=ast.Name(id='Any', ctx=ast.Load()), 
                                    simple=1) for name in all_names]
        
        elif isinstance(node, ast.AnnAssign):
            if self.in_global_scope:
                self.all_objs.add(node.target.id)
            return node
            return ast.AnnAssign(target = node.target,
                                 annotation=ast.Name(id='Any', ctx=ast.Load()),
                                 value = node.value,
                                 simple=1)         
        else:
            return super().generic_visit(node)
        #for, if, with stmt
        # elif hasattr(node, 'body') and isinstance(node.body, list):
        #     new_body = []
        #     for stmt in node.body:
        #         new_stmt = self.visit(stmt)
        #         if new_stmt:
        #             if isinstance(new_stmt, list):
        #                 new_body.extend(new_stmt)
        #             else:
        #                 new_body.append(new_stmt)
        #     if not new_body:
        #         new_body = [ast.Pass()]
        #     node.body = new_body
        #     if hasattr(node, 'orelse') and isinstance(node.orelse, list) and node.orelse:
        #         else_body = []
        #         for stmt in node.orelse:
        #             new_stmt = self.visit(stmt)
        #             if new_stmt:
        #                 if isinstance(new_stmt, list):
        #                     else_body.extend(new_stmt)
        #                 else:
        #                     else_body.append(new_stmt)
        #         if not else_body:
        #             else_body = [ast.Pass()]
        #         node.orelse = else_body
        #     return node
        # else:
        #     return super().generic_visit(node)
    
    def _process_function(self, node):
        # 处理所有参数
        for arg in node.args.posonlyargs:
            del arg.annotation
            # arg.default = None
        for arg in node.args.args:
            del arg.annotation
            # arg.default = None
        if node.args.vararg:
            del node.args.vararg.annotation
            # node.args.vararg.default = None
        for arg in node.args.kwonlyargs:
            del arg.annotation
            # arg.default = None
        if node.args.kwarg:
            del node.args.kwarg.annotation
            # node.args.kwarg.default = None
            
        if node.name not in ['__init__','__init_subclass__']:
            del node.returns
        # node.args.defaults = []
        # node.args.kw_defaults = [None for i in node.args.kw_defaults]
        # node.body = [ast.Expr(value=ast.Constant(Ellipsis))]
        # if self.in_class_scope and node.args.args and node.args.args[0].arg in ('self', 'cls'):
        #     del node.args.args[0].annotation
        decorator_list = []
        for decorator in node.decorator_list:
            if isinstance(decorator,ast.Name) and decorator.id in ['property', 'staticmethod'] \
                or isinstance(decorator,ast.Attribute) and decorator.attr in ['setter']:
                decorator_list.append(decorator)
        if isinstance(node, ast.FunctionDef):
            newNode = ast.FunctionDef(name = node.name, args = node.args, body = node.body,decorator_list=node.decorator_list, returns = node.returns)
        elif isinstance(node, ast.AsyncFunctionDef):
            newNode = ast.AsyncFunctionDef(name = node.name, args = node.args, body = node.body,decorator_list=node.decorator_list, returns = node.returns)
        return newNode
    
    def _process_class(self, node):
        new_body = []
        old_cls_scope = self.in_class_scope
        self.in_class_scope = True
        for stmt in node.body:
            new_stmt = self.visit(stmt)
            if new_stmt:
                if isinstance(new_stmt, list):
                    new_body.extend(new_stmt)
                else:
                    new_body.append(new_stmt)
        if not new_body:
            new_body = [ast.Pass()]
        new_bases = []
        for stmt in node.bases:
            if isinstance(stmt, ast.Name) and stmt.id in ['NamedTuple']:
                new_bases.append(stmt)
        node.body = new_body
        self.in_class_scope = old_cls_scope
        return node
        newNode = ast.ClassDef(name = node.name, bases = [], keywords = new_bases, body = new_body, decorator_list=[])
        return newNode

def ensure_any_import(tree):
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == 'typing':
            for alias in node.names:
                if alias.name == 'Any':
                    return tree, 0
    import_any = ast.ImportFrom(
        module='typing',
        names=[ast.alias(name='Any')],
        level=0
    )
    tree.body.insert(0, import_any)
    return tree, 1

def ensure_union_import(tree):
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == 'typing':
            for alias in node.names:
                if alias.name == 'Union':
                    return tree
    import_union = ast.ImportFrom(
        module='typing',
        names=[ast.alias(name='Union')],
        level=0
    )
    tree.body.insert(0, import_union)
    return tree

import tokenize
import io
def extract_comment(source_code):
    """记录源代码中的注释（行号与内容）"""
    comments = {}
    if not source_code.strip():
        return comments

    try:
        tokens = tokenize.generate_tokens(io.StringIO(source_code).readline)
        for token in tokens:
            if token.type == tokenize.COMMENT:
                # 记录格式: {行号: 注释内容}
                comments[token.start[0]] = token.string
    except tokenize.TokenError:
        pass  # 忽略词法分析错误
    return comments


def restore_comments(new_code, comments, added_lines=[]):
    """将注释恢复到新代码的对应行号，考虑新增行数
    
    Args:
        new_code: 新生成的代码字符串
        comments: 记录的注释字典 {行号: 注释内容}
        added_lines: 新增行列表 [(基准行号, 新增行数)]
    """
    if not new_code.strip():
        return new_code

    lines = new_code.splitlines()
    # 创建行号映射字典
    # 如果没有新增行，直接创建1:1的映射
    if not added_lines:
        new_comment_dict = comments
    else:
        # 按基准行号排序
        added_lines.sort()
        
        new_comment_dict = {}
        for line_num, comment in comments.items():
            for base_line, added_lines_num in added_lines:
                if line_num > base_line:
                    line_num += added_lines_num
            new_comment_dict[line_num] = comment
    
    # 使用行号映射恢复注释
    for line_num, comment in new_comment_dict.items():
        if line_num <= len(lines):
            target_line = lines[line_num - 1]
            if target_line.strip():  # 非空行：附加在行尾
                lines[line_num - 1] = f"{target_line.rstrip()}  {comment.lstrip()}"
            else:  # 空行：直接替换为注释
                lines[line_num - 1] = comment
    
    return '\n'.join(lines)

def generate_stub(source_file, stub_file):
    source = Path(source_file).read_text()
    comments_dict = extract_comment(source)
    tree = ast.parse(source)
    transformer = TypeAnnotationTransformer()
    tree = transformer.visit(tree)
    all_objs = transformer.all_objs
    tree = ast.parse(source)
    tree.body.append(
        ast.Assign(
            targets = [ast.Name(id='__all__', ctx=ast.Store())],
            value = ast.List(elts=[ast.Constant(value=obj) for obj in all_objs], ctx=ast.Load()),
            simple=1
        )
    )
    # for asname, importedName in transformer.import_asName.items():
    #     assign_node = ast.Assign(targets=[ast.Name(id=asname, ctx=ast.Store())], value=ast.Name(id=importedName, ctx = ast.Load()))
    #     tree.body.append(assign_node)
    tree = ast.fix_missing_locations(tree)
    stub_code = ast.unparse(tree)
    stub_code = restore_comments(stub_code, comments_dict)
    Path(stub_file).write_text(stub_code + '\n')
        
def process_project(project):
    rootDir:Path = untypedProjectsDir / project
    destDir:Path = prototype_Path / project
    
    if os.path.exists(destDir):
        shutil.rmtree(destDir)  
    shutil.copytree(rootDir, destDir)
    for file in destDir.rglob('*.py'):
        generate_stub(file, file)

def extract_msg_error_detail(msg):
    #extract the error trigger snippet from
    class msg_locator(ast.NodeVisitor):
        def __init__(self, lineno):
            #input_info
            self.lineno = lineno
            #output_info
            self.parent_function = None
            self.location_stmt = None
        
        def generic_visit(self, node):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and self.lineno >= node.lineno and self.lineno <= node.end_lineno:
                    self.parent_function = node
            if isinstance(node, (ast.stmt, ast.expr)) and self.lineno == node.lineno and self.location_stmt is None:
                self.location_stmt = node
            return super().generic_visit(node)
    
    file_path = msg['file_path']
    line_number = msg['line_number']
    description = msg['description']
    full_path = os.path.join(os.getcwd(),file_path)
    ast_tree = ast.parse(open(full_path).read())
    ml = msg_locator(line_number)
    ml.visit(ast_tree)
    if not ml.location_stmt:
        raise Exception(f"Cannot find the location of the error message: {msg}")
    loc_stmt = ast.unparse(ml.location_stmt).splitlines()[-1]
    if ml.parent_function:
        errorMsg = ast.unparse(ml.parent_function)
    else:
        errorMsg = loc_stmt
    errorMsg = errorMsg.replace(loc_stmt, f"{loc_stmt} # {description}")
    if len(errorMsg) > 1000:
        logger.warning(f"Error message is too long: {errorMsg}")
        errorMsg = f"{loc_stmt} # {description}"
        if ml.parent_function:
            errorMsg += f"\n in function {ml.parent_function.name}"
    return errorMsg

def extract_msg_error_info(msg):
    class msg_locator(ast.NodeVisitor):
        def __init__(self, lineno):
            #input_info
            self.lineno = lineno
            #output_info
            self.parent_function = None
            self.location_stmt = None
        
        def visit_FunctionDef(self, node):
            if self.lineno >= node.lineno and self.lineno <= node.end_lineno:
                self.parent_function = node
            return super().generic_visit(node)
        
        def generic_visit(self, node):
            if isinstance(node, (ast.stmt, ast.expr)) and self.lineno == node.lineno and self.location_stmt is None:
                self.location_stmt = node
            return super().generic_visit(node)
    
    file_path = msg['file_path']
    line_number = msg['line_number']
    description = msg['description']
    full_path = os.path.join(os.getcwd(),file_path)
    ast_tree = ast.parse(open(full_path).read())
    ml = msg_locator(line_number)
    ml.visit(ast_tree)
    if not ml.location_stmt:
        raise Exception(f"Cannot find the location of the error message: {msg}")
    loc_stmt = ast.unparse(ml.location_stmt).splitlines()[-1]
    if ml.parent_function:
        errorMsg = ast.unparse(ml.parent_function)
    else:
        errorMsg = loc_stmt
    errorMsg = errorMsg.replace(loc_stmt, f"{loc_stmt} # {description}")
    return errorMsg
   
def extract_mypy_errors(error_lines):
    # 正则表达式模式，用于匹配清理后的错误行
    pattern = re.compile(
        r'^(?P<filePath>[^:]+):'
        r'(?P<lineNumber>\d+):\s*'
        r'(?P<errorType>[\w]+):\s*'
        r'(?P<description>.*)\s+\['
        r'(?P<errorCode>[\w-]+)\]'
    )
    extracted = []
    for line in error_lines:
        # 去除ANSI转义字符（颜色代码）
        cleaned_line = re.sub(r'\x1b$$[0-9;]*m', '', line)
        # 匹配错误信息
        match = pattern.match(cleaned_line)
        if match:
            file_path = match.group('filePath')
            line_number = match.group('lineNumber')
            error_type = match.group('errorType').lower()  # 统一小写处理
            description = match.group('description').strip()
            error_code = match.group('errorCode')
            extracted.append({
                'file_path': file_path,
                'line_number': int(line_number),
                'error_type': error_type,
                'description': description,
                'error_code': error_code
            })
    if not extracted:
        return [] #error_lines
    else:
        return extracted

def run_mypy(file_path):
    
    result = subprocess.run(
        ["python","-m","mypy", file_path, "--disable-error-code", "var-annotated", "--disable-error-code", "assignment", "--disable-error-code", "has-type", "--ignore-missing-imports", "--check-untyped-defs", "--no-incremental","--follow-imports=silent"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={"MYPYPATH":file_path}.update(os.environ),
        text=True
    )
    logger.debug(f"mypy result:\n {result.stdout}")
    if result.stderr:
        logger.error(f"mypy failed with return code {result.returncode}")
        logger.error(f"mypy output:\n {result.stdout}")
        logger.error(f"mypy error:\n {result.stderr}")
        raise RuntimeError("mypy failed")
    errorMsg = result.stdout.splitlines()
    extracted_errMsg = extract_mypy_errors(errorMsg)
    return extracted_errMsg

def run_mypy_strict(file_path):
    result = subprocess.run(
        ["python","-m","mypy", file_path, "--ignore-missing-imports", "--check-untyped-defs", "--no-incremental","--follow-imports=silent"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={"MYPYPATH":file_path}.update(os.environ),
        text=True
    )
    logger.debug(f"mypy result:\n {result.stdout}")
    if result.stderr:
        logger.error(f"mypy failed with return code {result.returncode}")
        logger.error(f"mypy output:\n {result.stdout}")
        logger.error(f"mypy error:\n {result.stderr}")
        raise RuntimeError("mypy failed")
    errorMsg = result.stdout.splitlines()
    extracted_errMsg = extract_mypy_errors(errorMsg)
    return extracted_errMsg

def shortest_detail_err(errMsg):
    errMsg_details = [extract_msg_error_detail(msg) for msg in errMsg]
    shortest_errMsg = min(errMsg_details, key=len)
    
    return shortest_errMsg

def validate_project(project):
    destDir:Path = prototype_Path / project
    extracted_errMsg = run_mypy(destDir)
    errMsg_details = [msg for msg in extracted_errMsg]
    logger.error(f"detailed error:\n {errMsg_details}")
    return errMsg_details

    

class ReExporter(ast.NodeTransformer):
    def __init__(self, attrs):
        self.doneAttrs = {attr:False for attr in attrs}
    def generic_visit(self, node):
        if isinstance(node, (ast.Import,ast.ImportFrom)):
            for alias in node.names:
                if alias.asname and alias.asname in self.doneAttrs:
                    newNode = ast.Assign(targets=[ast.Name(id=alias.asname, ctx=ast.Store())], value=ast.Name(id=alias.name, ctx=ast.Load()))
                    self.doneAttrs[alias.asname] = True
                    alias.asname = None
                    return [node, newNode]
            return node
        else:
            return super().generic_visit(node)

if __name__ == '__main__':
    for project in projects:
        print(project)
        process_project(project)
        print(validate_project(project))
        