from type_llm.utils.config import projects, untypedProjectsDir, Scope_Path
import ast
import os
from type_llm.utils import silent_JDump
from type_llm.utils.log_manager import setup_logger
import importlib

logger = setup_logger()
class Component_Collector(ast.NodeVisitor):
    def __init__(self, module_name, all_modules):
        self.symbol_map = {}
        self.module_name = module_name
        self.sub_symbol_map = {}
        self.all_modules = all_modules
    
    def get_full_module_name(self, module_name):
        base_name = '.'.join(self.module_name.split('.')[:-1])
        while True:
            if base_name + '.' + module_name in self.all_modules:
                return base_name + '.' + module_name
            elif not base_name:
                break
            else:
                base_name = '.'.join(base_name.split('.')[:-1])
        return None
     
    def visit_Import(self, node):
        symbol_map = {}
        for alias in node.names:   
            full_name = self.get_full_module_name(alias.name)
            if full_name is None:
                logger.warning(f"Module {alias.name} not found in {self.module_name}")
                continue
            if alias.asname:
                symbol_map[alias.asname] = alias.name
            else:
                symbol_map[alias.name] = alias.name               
        return symbol_map
    
    def visit_ImportFrom(self, node):
        symbol_map = {}
        if node.level:
            base_module = '.'.join(self.module_name.split('.')[:-node.level])
            if node.module:
                base_module += '.'+node.module
        else:
            full_name = self.get_full_module_name(node.module)
            if full_name is None:
                logger.warning(f"Module {node.module} not found in {self.module_name}")
                return {}
            else:
                base_module = full_name            
        for alias in node.names:
            if alias.asname:
                symbol_map[alias.asname] = base_module + '.' + alias.name
            else:
                symbol_map[alias.name] = base_module + '.' + alias.name
        return symbol_map
    
    def visit_Assign(self, node):
        symbol_map = {}
        for target in node.targets:
            if isinstance(target, ast.Name):
                symbol_map[target.id] = self.module_name + '.' + target.id
            elif isinstance(target, ast.Tuple):
                for name in target.elts:
                    if isinstance(name, ast.Name):
                        symbol_map[name.id] = self.module_name + '.' + name.id
        return symbol_map
    
    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Name):
            return {node.target.id: self.module_name + '.' + node.target.id}
        else:
            return {}
    
    def visit_FunctionDef(self, node):
        for stmt in node.body:
            if isinstance(stmt, ast.ClassDef):
                old_module = self.module_name
                self.module_name = self.module_name + '.' + node.name
                self.visit(stmt)
                self.module_name = old_module
        return {node.name: self.module_name + '.' + node.name}
    
    def visit_AsyncFunctionDef(self, node):
        return self.visit_FunctionDef(node)
    
    def visit_ClassDef(self, node):
        symbol_map = {node.name: self.module_name + '.' + node.name}
        local_symbol_map ={}
        old_module = self.module_name
        self.module_name += '.' + node.name
        for stmt in node.body:
            stmt_symbol_map = self.visit(stmt)
            if stmt_symbol_map:
                local_symbol_map.update(stmt_symbol_map)
        self.module_name = old_module
        self.sub_symbol_map[self.module_name+'.'+node.name] = local_symbol_map
        return symbol_map
    
    def generic_visit(self, node):
        symbol_map = {}
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        stmt_dict = self.visit(item)
                        if stmt_dict:
                            symbol_map.update(stmt_dict)
            elif isinstance(value, ast.AST):
                stmt_dict = self.visit(value)
                if stmt_dict:
                    symbol_map.update(stmt_dict)
        return symbol_map
    
    def visit_Module(self, node):
        for stmt in node.body:
            stmt_dict = self.visit(stmt)
            if stmt_dict:
                self.symbol_map.update(stmt_dict)
        return self.symbol_map

def collect_all_module(dir, module):
    module_list = []
    for file in os.listdir(dir):
        full_path = os.path.join(dir, file)
        if os.path.isfile(full_path) and file.endswith('.py'):
            file_module = module+'.'+file[:-3]
            module_list.append(file_module)
        elif os.path.isdir(full_path):
            file_module = module+'.'+file
            module_list.append(file_module)
            module_list.extend(collect_all_module(full_path, module+'.'+file))
    return module_list
  
def build_scope(dir, module, all_modules):
    scope_dict = {}
    for file in os.listdir(dir):
        full_path = os.path.join(dir, file)
        if os.path.isfile(full_path) and file.endswith('.py'):
            file_module = module+'.'+file[:-3]
            with open(full_path, 'r') as f:
                tree = ast.parse(f.read())
            collector = Component_Collector(file_module, all_modules)
            collector.visit(tree)
            scope_dict[file_module] = collector.symbol_map
            scope_dict.update(collector.sub_symbol_map)
        elif os.path.isdir(full_path):
            scope_dict.update(build_scope(full_path, module+'.'+file,all_modules))
    return scope_dict
    
if __name__ == "__main__":
    for project in projects:
        root_dir = os.path.join(untypedProjectsDir, project)
        all_modules = collect_all_module(root_dir, project)
        scope_dict = build_scope(root_dir, project, all_modules)
        
        silent_JDump(scope_dict, Scope_Path / f"{project}.json")
    