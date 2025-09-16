import json
import os
import re
import ast

def silent_JDump(obj, filePath):
    parent = filePath.parent
    if not os.path.exists(parent):
        parent.mkdir(parents=True)
    with open(filePath, 'w') as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)

def silent_Write(obj, filePath):
    parent = filePath.parent
    if not os.path.exists(parent):
        parent.mkdir(parents=True)
    with open(filePath, 'w') as f:
        f.write(obj)

def silent_Append(obj, filePath):
    parent = filePath.parent
    if not os.path.exists(parent):
        parent.mkdir(parents=True)
    with open(filePath, 'a') as f:
        f.write(obj)

class funcTransformer(ast.NodeTransformer):
    def func_trans(self, node:ast.FunctionDef|ast.AsyncFunctionDef):
        pass
    
    def visit_FunctionDef(self, node):
        return self.func_trans(node)
        
    def visit_AsyncFunctionDef(self, node):
        return self.func_trans(node)

class funcVisitor(ast.NodeVisitor):
    def func_visit(self, node:ast.FunctionDef|ast.AsyncFunctionDef):
        pass
    
    def visit_FunctionDef(self, node):
        self.func_visit(node)
        
    def visit_AsyncFunctionDef(self, node):
        self.func_visit(node)  
   

def extract_code(msg):
    try:
        stub = re.findall( r'```python(.*?)```', msg, re.DOTALL)[-1]
    except:
        return -1, "No Python Code Found"
    try:
        ast.parse(stub)
    except:
        return -2, "Sematic Error Exists"
    return 0,stub

def json2stub(jsonMsg:str):
    if jsonMsg is None:
        return None
    data = json.loads(jsonMsg)
    if data == 'fail':
        return None
    msg = data[1]['content']
    code, stub = extract_code(msg)
    if code == 0:
        return stub
    else:
        return None

def is_valid_msg(msg):
    return extract_code(msg)[0] == 0
