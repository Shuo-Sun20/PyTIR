from type_llm.utils import PyAnalyzer_Enum as PE, PyAnalyzer_Model as PM, silent_Write
from type_llm.utils.log_manager import setup_logger
from typing import Dict, Set, List, Optional, Tuple
from type_llm.methods.full_LARRY import Entity_Graph as EG
from pydantic import BaseModel
from type_llm.utils.config import projects, Raw_PyAnalyzer, prototype_Path, EntityGraph_Path
import ast
import re

logger = setup_logger()

def is_compose_change_stmt(stmt:str, def_name:str):
    keyword_list = ['append','insert','setdefault','add']
    for keyword in keyword_list:
        if def_name+'.'+keyword in stmt:
            return True
    escaped_name = re.escape(def_name)
    pattern = re.compile(
        r'\s+' + escaped_name + r'\[\s*[^][]+?\s*\]\s*=[^=]',
        re.IGNORECASE
    )
    if re.search(pattern, stmt):
        return True
    return False

def is_valid_file(file:str):
    full_Path = prototype_Path / file
    return full_Path.exists() and full_Path.is_file()

def attr_extract(fullName:str):
    astNode = ast.parse(fullName).body[0].value
    if isinstance(astNode, ast.Attribute):
        return astNode.attr
    else:
        raise Exception(f"{fullName}---{ast.dump(astNode)} is Not an attribute")

def valid_snippet(stmt, node_name):
    def contain(node, name):
        if isinstance(node, ast.Name):
            return node.id == name
        elif isinstance(node, ast.Attribute):
            return "self." in ast.unparse(node) and node.attr == name
        elif isinstance(node, ast.Subscript):
            return contain(node.value, name)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for item in node.elts:
                if contain(item, name):
                    return True
            return False
        else:
            return False
    
    stmt = ast.parse(stmt).body[0]
    attr_name = node_name.split('.')[-1]
    if isinstance(stmt, ast.Assign):
        for target in stmt.targets:
            if contain(target, attr_name):
                return True
        return False
    elif isinstance(stmt, ast.AugAssign):
        return contain(stmt.target, attr_name)
    elif isinstance(stmt, ast.AnnAssign):
        return contain(stmt.target, attr_name)
    else:
        return True

def get_def_stmt_type(stmt:Optional[str]):
    typeMap = {
        "Assign": EG.Entity_Type.var,
        "AugAssign": EG.Entity_Type.var,
        "AnnAssign": EG.Entity_Type.var,
        "FunctionDef": EG.Entity_Type.func,
        "AsyncFunctionDef": EG.Entity_Type.func,
        "ClassDef": EG.Entity_Type.class_
    }
    if not stmt:
        return "Invalid"
    else:
        astNode = ast.parse(stmt).body[0]
        clsName = astNode.__class__.__name__
        if clsName in typeMap:
            return typeMap[astNode.__class__.__name__]
        else:
            return "Invalid"

def is_local(srcName:str, destName:str):
    return '.'.join(destName.split('.')[:-1]) == srcName

def is_valid_entity(def_location:EG.Stmt_Location):
    
    class Line_Filler(ast.NodeVisitor):
        def __init__(self, line):
            self.line = line
            self.type = None
            
        def generic_visit(self, node):
            if isinstance(node, (ast.stmt, ast.expr)) and node.lineno == self.line:
                self.type = get_def_stmt_type(ast.unparse(node))
            else:
                return super().generic_visit(node)
    
    full_file = prototype_Path / def_location.file
    with open(full_file, 'r') as f:
        tree = ast.parse(f.read())
    lf = Line_Filler(def_location.line)
    lf.visit(tree)
    return lf.type != "Invalid"

def from_pyanalyzer(pa:PM.PyAnalyzerResult, project:str):
    #helper functions
    #根据PyAnalyzer中每个Variable的location，生成Name->Loc，
    def create_def_location_dict(id_raw_dict: Dict[int, PM.PyAnalyzerVariableResult], cell_list: List[PM.PyAnalyzerCellResult]):
        name_location_map: Dict[str, Set[EG.Stmt_Location]] = {}
        def_name_dict: Dict[str,Tuple[str, str]] = {}
        waiting_dict:Dict[int, PM.PyAnalyzerVariableResult] = {}
        for id, var in id_raw_dict.items():
            if var.category not in PE.Internal_EntKinds:
                logger.debug(f"{var.qualified_name} is not an internal entity")
            else:
                qualified_name = var.qualified_name
                if (not var.file) or (var.location.start_line < 1):
                    if var.category == PE.EntKind.UnresolvedAttr:
                        waiting_dict[id] = var
                    else:
                        raise Exception(f"Var{var.id}--{var.qualified_name} has no location information")
                else:
                    location = EG.Stmt_Location(file=var.file, line=var.location.start_line)
                    if is_valid_entity(location):
                        name_location_map.setdefault(qualified_name,set()).add(location)
                        related_name = qualified_name.split('.')[-1]
                        def_name_dict[qualified_name] = (var.file, related_name)
        for cell in cell_list:
            if cell.values.kind in PE.Def_RefKinds and cell.src in id_raw_dict:
                if cell.dest in waiting_dict:
                    qualified_name = waiting_dict.pop(cell.dest).qualified_name
                    def_name_dict[qualified_name] = (id_raw_dict[cell.src].file, qualified_name.split('.')[-1])
                    location = EG.Stmt_Location(file=id_raw_dict[cell.src].file, line=cell.location.start_line)
                    name_location_map.setdefault(qualified_name,set()).add(location)
        return name_location_map,def_name_dict
    
    def classify_cells(cell_list:List[PM.PyAnalyzerCellResult], node_dict: Dict[int, PM.PyAnalyzerVariableResult]):
        inheritance_cells: List[PM.PyAnalyzerCellResult] = []
        def_cells: List[PM.PyAnalyzerCellResult] = []
        ref_cells: List[PM.PyAnalyzerCellResult] = []
        param_localVars: Set[str] = set()
        nest_dict:Dict[int,int] = {}
        for cell in cell_list:
            #过滤无效边
            if cell.values.kind in PE.Ignored_RefKinds:
                logger.debug(f"过滤掉{cell.values.kind}类型变量{cell.src}到{cell.dest}的引用")
            elif cell.src not in node_dict or cell.dest not in node_dict:
                logger.debug(f"过滤掉{cell.src}或{cell.dest}不在变量列表中的引用")
            elif cell.location.start_line < 1:
                logger.debug(f"过滤掉{cell.src}到{cell.dest}的引用，因为起始行小于1")
            elif not is_valid_file(node_dict[cell.src].file) :
                logger.debug(f"过滤掉{cell.src}到{cell.dest}的引用，因为起始文件{node_dict[cell.src].file}无效")
            elif cell.values.kind == PE.RefKind.InheritKind:
                inheritance_cells.append(cell)
            else:
                src_var = node_dict[cell.src]
                dest_var = node_dict[cell.dest]
                if src_var.category in [PE.EntKind.Module, PE.EntKind.Class]: #statements in scopes
                    if cell.values.kind in PE.Def_RefKinds:
                        def_cells.append(cell)
                    elif cell.values.kind in PE.Use_RefKinds:
                        ref_cells.append(cell)
                    else:
                        raise ValueError(f"Unexpected ref kind {cell.values.kind} in {cell.src} to {cell.dest}")
                elif src_var.category == PE.EntKind.Function: #statements in functions
                    if is_local(src_var.qualified_name, dest_var.qualified_name):
                        logger.debug(f"暂时不考虑函数{src_var.qualified_name}与参数、局部变量{dest_var.qualified_name}之间的关系")
                        param_localVars.add(dest_var.qualified_name)
                        if dest_var.category == PE.EntKind.Function:
                            nest_dict[cell.dest] = cell.src
                    elif cell.values.kind in PE.Def_RefKinds: #函数中定义外部变量
                        if dest_var.category in (PE.EntKind.ClassAttr, PE.EntKind.UnresolvedAttr):
                            def_cells.append(cell)
                        else:
                            logger.warning(f"函数{src_var.qualified_name}中定义了外部变量{dest_var.qualified_name}")
                    elif cell.values.kind in PE.Use_RefKinds:
                        ref_cells.append(cell)
                    else:
                        raise ValueError(f"Unexpected ref kind {cell.values.kind} in {cell.src} to {cell.dest}")
                else:
                    raise ValueError(f"Unexpected src category {src_var.category} in {cell.src} to {cell.dest}")
        return inheritance_cells, def_cells, ref_cells, param_localVars,nest_dict
    
    def loc2Snippet(loc_list:Set[EG.Stmt_Location], project:str)->Dict[EG.Stmt_Location, EG.Snippet]:
        file_line_dict:Dict[str, Dict[int, Optional[EG.Snippet]]] = {}
        for loc in loc_list:
            file_line_dict.setdefault(loc.file, {}).setdefault(loc.line, None)
        root_dir = prototype_Path / project
        current_module = project
        EG.extract_snippet_from_dir(root_dir, current_module, file_line_dict)
        results:Dict[EG.Stmt_Location, EG.Snippet] = {}
        for file, line_dict in file_line_dict.items():
            for line, snippet in line_dict.items():
                if snippet is not None:
                    results[EG.Stmt_Location(file=file, line=line)] = snippet
                else:
                    logger.warning(f"Snippet not found for {file}:{line}")
        return results
    
    def construct_dependency(dependency_dict:Dict[EG.Snippet_Id, Set[Tuple[str, EG.Stmt_Location]]], location_use_nodes:Dict[EG.Stmt_Location, Set[str]], node_def_locations:Dict[str, Set[EG.Stmt_Location]]):
        snippet_dep:Dict[EG.Snippet_Id, Set[str]] = dependency_dict
        for entity_name, snippet_locations in node_def_locations.items():
            for snippet_location in snippet_locations:
                snippet_ID = EG.Snippet_Id(defined_name=entity_name, location=snippet_location)
                snippet_dep.setdefault(snippet_ID,set()).update(location_use_nodes.get(snippet_location, set()))
        return snippet_dep
    
    def construct_graph(project: str,
                        snippet_EN_dep:Dict[EG.Snippet_Id, Set[Tuple[str, EG.Stmt_Location]]], 
                        node_def_locations:Dict[str, Set[EG.Stmt_Location]], 
                        file_line_stmt_dict:Dict[EG.Stmt_Location, EG.Snippet], 
                        supper_snippet_dict:Dict[EG.Snippet_Id, EG.Snippet_Id],
                        def_name_dict:Dict[str,Tuple[str, str]],
                        append_dep_dict:Dict[str,str],
                        supper_cls_dict:Dict[str, Set[str]]
                        ):
        
        snippet_dict:Dict[EG.Snippet_Id, EG.Snippet_Unit] = {}
        node_dict : dict[str, EG.Entity_Node] = {}
        for node_name, snippet_locations in node_def_locations.items():
            snippet_set = set()
            dependency = set()
            all_possible_types:Set[EG.Entity_Type] = set()
            for location in snippet_locations:
                snippet_ID = EG.Snippet_Id(defined_name=node_name, location=location)
                stmt = file_line_stmt_dict[location].snippet
                stmt_type = get_def_stmt_type(stmt)
                if not valid_snippet(stmt, node_name):
                    continue
                if stmt_type == 'Invalid':
                    stmt_type = EG.Entity_Type.var
                all_possible_types.add( stmt_type )
                if snippet_ID in supper_snippet_dict and not isinstance(ast.parse(stmt).body[0], ast.AugAssign):
                    supper_ID = supper_snippet_dict[snippet_ID]
                    new_SU = EG.Snippet_Unit(ID = snippet_ID, snippet=stmt, supper_snippet=supper_ID)
                    dependency.add(supper_ID.defined_name)
                else:
                    dep_set = snippet_EN_dep.get(snippet_ID, set())
                    clear_dep_set = set()
                    for dep in dep_set:
                        if dep[0] != snippet_ID.defined_name:
                            clear_dep_set.add((dep[0],  file_line_stmt_dict[dep[1]].snippet))
                    new_SU = EG.Snippet_Unit(ID = snippet_ID, snippet=stmt, dependency=clear_dep_set)
                    dependency.update([v[0] for v in clear_dep_set])
                snippet_dict[snippet_ID] = new_SU
                snippet_set.add(new_SU)                
            if len(all_possible_types) == 1:
                node_type = list(all_possible_types)[0]
                append_dep = append_dep_dict[node_name] if node_name in append_dep_dict else set()
                node_dict[node_name] = EG.Entity_Node(qualified_name=node_name, 
                                                      category=node_type, 
                                                      def_file= def_name_dict[node_name][0],
                                                      def_name= def_name_dict[node_name][1],
                                                      dependency= dependency, 
                                                      snippet_set=snippet_set,
                                                      appended_dependency = append_dep)
            
        return EG.Entity_Graph(project =project, node_dict=node_dict, snippet_dict=snippet_dict, supper_cls_dict = supper_cls_dict)
    
    def resolve_inheritance(inherited_class_dict:Dict[str, Set[str]], 
                            class_attr_dict:Dict[str, Dict[str,Set[str]]], 
                            attr_class_dict:Dict[str, Set[str]]):
        #基于类间继承关系，补全class-attribute关系 和 attr-class倒排表
        while inherited_class_dict:
            #获取所有根类
            root_Classes = set(inherited_class_dict.keys())
            for inherited_cs in inherited_class_dict.values():
                root_Classes -= inherited_cs
            #处理根类的子类
            for root_class in root_Classes:
                child_set = inherited_class_dict.pop(root_class)
                if root_class not in class_attr_dict:
                    logger.info(f"根类{root_class}无定义")
                else:
                    #补全子类的属性依赖
                    root_attrDict = class_attr_dict.get(root_class, set())
                    for cls_id in child_set:
                        child_attrSet = class_attr_dict.get(cls_id, set())
                        for attr,attr_id_set in root_attrDict.items():
                            #如果子类没有该属性，则添加
                            if attr not in child_attrSet:
                                class_attr_dict.setdefault(cls_id,{})[attr] = attr_id_set
                                attr_class_dict[attr].add(cls_id)
    
    def clear_unnecessary_supper_snippet_dict(supper_snippet_dict:Dict[EG.Snippet_Id, EG.Snippet_Id],
                                              id_raw_dict: Dict[int, PM.PyAnalyzerVariableResult],
                                              all_cells: List[PM.PyAnalyzerCellResult]):
        function_use_location:Dict[str,Set[EG.Stmt_Location]] = {}
        for cell in all_cells:
            #过滤无效边
            if cell.values.kind in PE.Ignored_RefKinds \
                or cell.src not in id_raw_dict \
                or cell.location.start_line < 1 \
                or not is_valid_file(id_raw_dict[cell.src].file)  \
                or cell.values.kind == PE.RefKind.InheritKind :
                    continue        
            elif cell.values.kind in PE.Use_RefKinds and id_raw_dict[cell.src].category == PE.EntKind.Function:
                function_use_location.setdefault(id_raw_dict[cell.src].qualified_name, set()).add(EG.Stmt_Location(file = id_raw_dict[cell.src].file, line = cell.location.start_line))
        necessary_snippet_dict:Dict[EG.Snippet_Id, EG.Snippet_Id] = {}
        for snippet_id, super_snippet_id in supper_snippet_dict.items():
            if snippet_id.location in function_use_location.get(super_snippet_id.defined_name,set()):
                necessary_snippet_dict[snippet_id] = super_snippet_id
            else:
                logger.debug(f"Dependency from {snippet_id} to {super_snippet_id} is not necessary")
        return necessary_snippet_dict
    
    #basic Var&Cell Info        
    id_raw_dict = {var.id:var for var in pa.variables if var.category not in PE.Ignored_EntKinds and not var.qualified_name.endswith('.__all__')}
    node_def_locations, def_name_dict = create_def_location_dict(id_raw_dict, pa.cells) # create node dict and initial def locations
    inheritance_cells, def_cells, ref_cells,param_localVars,nest_dict = classify_cells(pa.cells, id_raw_dict) # classify cells
    #函数的参数和局部变量不视为实体
    for param in param_localVars:
        if param in node_def_locations:
            node_def_locations.pop(param)
    #函数的嵌套关系
    while(nest_dict):
        leaves = [k for k,v in nest_dict.items() if v not in nest_dict]
        for leaf in leaves:
            parent = nest_dict.pop(leaf)
            for cell in def_cells:
                if cell.src == leaf:
                    cell.src = parent
            for cell in ref_cells:
                if cell.src == leaf:
                    cell.src = parent
        
    #Dependency 
    location_use_nodes:Dict[EG.Stmt_Location, Set[str]] = {} 
    dependency_dict:Dict[EG.Snippet_Id, Set[Tuple[str, EG.Stmt_Location]]] = {} 
    supper_snippet_dict:Dict[EG.Snippet_Id, EG.Snippet_Id] = {} 
    
    #possible type related info
    class_attr_dict:Dict[str, Dict[str,Set[str]]] = {} # class attr dict
    attr_class_dict:Dict[str, Set[str]] = {} # attr class dict
    base_sub_dict:Dict[str, Set[str]] = {} # base sub dict
    supper_cls_dict:Dict[str, Set[str]] = {}
    for cell in inheritance_cells:
        base_sub_dict.setdefault(id_raw_dict[cell.dest].qualified_name, set()).add(id_raw_dict[cell.src].qualified_name)
        supper_cls_dict.setdefault(id_raw_dict[cell.src].qualified_name, set()).add(id_raw_dict[cell.dest].qualified_name)
    
    changed = True
    while changed:
        changed = False
        for sub_cls,super_cls_set in supper_cls_dict.items():
            new_super_cls_set = set(super_cls_set)
            for supper_cls in super_cls_set:
                if supper_cls in supper_cls_dict:
                    new_super_cls_set.update(supper_cls_dict[supper_cls])
            if new_super_cls_set != super_cls_set:
                supper_cls_dict[sub_cls] = new_super_cls_set
                changed = True
        
    #收集相关location
    related_locs = set()
    for cell in def_cells:
        #PA_Part
        src_var = id_raw_dict[cell.src]
        dest_var = id_raw_dict[cell.dest]
        #EG_Part
        defined_name = dest_var.qualified_name
        location = EG.Stmt_Location(file = src_var.file, line = cell.location.start_line)
        #Analyze Cell
        if defined_name not in node_def_locations:
            logger.debug(f"{defined_name} is not an entity")
            continue
        else:
            related_locs.add(location)
    for cell in ref_cells:
        #PA_Part
        src_var = id_raw_dict[cell.src]
        dest_var = id_raw_dict[cell.dest]
        #EG_Part
        used_name = dest_var.qualified_name
        location = EG.Stmt_Location(file = src_var.file, line = cell.location.start_line)
        if src_var.category in PE.Scope_EntKinds:
            related_locs.add(location)
    
    file_line_stmt_dict = loc2Snippet(related_locs,project) #location->stmt Dict\
        
    for cell in def_cells:
        #PA_Part
        src_var = id_raw_dict[cell.src]
        dest_var = id_raw_dict[cell.dest]
        #EG_Part
        defined_name = dest_var.qualified_name
        location = EG.Stmt_Location(file = src_var.file, line = cell.location.start_line)
        #Analyze Cell
        if defined_name not in node_def_locations:
            logger.debug(f"{defined_name} is not an entity")
            continue
        
        snippet = file_line_stmt_dict[location]
        astNode = ast.parse(snippet.snippet).body[0]
        expect_names = []
        local_names = set()
        if isinstance(astNode, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)): 
            expect_names.append(snippet.scope.current_scope)   #函数定义所在行可以定义函数名及参数，但是我们不把参数视为实体
        elif isinstance(astNode, ast.Assign):
            for target in astNode.targets:
                if isinstance(target, (ast.Name, ast.Attribute)):
                    name,code = EG.get_defined_names(target, snippet.scope)
                    if code == 0:
                        expect_names.append(name)
                    elif code == 1:
                        local_names.add(name)
                elif isinstance(target, ast.Tuple):
                    for name in target.elts:
                        if isinstance(name, (ast.Name, ast.Attribute)):
                            name,code = EG.get_defined_names(name, snippet.scope)
                            if code == 0:
                                expect_names.append(name)
                            elif code == 1:
                                local_names.add(name)
        elif isinstance(astNode, (ast.AnnAssign,ast.AugAssign)):
            name,code = EG.get_defined_names(astNode.target, snippet.scope)
            if code == 0:
                expect_names.append(name)
            elif code == 1:
                local_names.add(name)
        else:
            continue
        
        for defined_name in expect_names:
            if defined_name not in node_def_locations:
                continue
            node_def_locations.setdefault(defined_name, set()).add(location)
            # function中定义或赋值attr, supper_snippet依赖
            if src_var.category == PE.EntKind.Function: 
                src_snippet_id = EG.Snippet_Id(defined_name=src_var.qualified_name,
                                            location= EG.Stmt_Location(
                                            file = src_var.file,
                                            line = src_var.location.start_line
                                            ))
                dest_snippet_id = EG.Snippet_Id(defined_name=defined_name,
                                            location=location)
                if dest_snippet_id in supper_snippet_dict and supper_snippet_dict[dest_snippet_id] != src_snippet_id:
                    raise ValueError(f"Multiple supper_snippet for {dest_snippet_id}: {supper_snippet_dict[dest_snippet_id]} and {src_snippet_id}")
                else:
                    supper_snippet_dict[dest_snippet_id] = src_snippet_id
            # class中定义或赋值attr/method, 记录 class-attr关系
            elif src_var.category == PE.EntKind.Class: 
                cls_name = src_var.qualified_name
                attrName = attr_extract(dest_var.qualified_name)
                class_attr_dict.setdefault(cls_name, {}).setdefault(attrName, set()).add(defined_name)
                attr_class_dict.setdefault(attrName, set()).add(cls_name)
            elif src_var.category == PE.EntKind.Module: # module中定义或赋值class/function/Var
                pass
            else:
                raise ValueError(f"Unexpected src category {src_var.category} in {cell.src} to {cell.dest}")
    
    append_dep_dict:Dict[str, str] = {}
    for cell in ref_cells:
        #PA_Part
        src_var = id_raw_dict[cell.src]
        dest_var = id_raw_dict[cell.dest]
        #EG_Part
        used_name = dest_var.qualified_name
        location = EG.Stmt_Location(file = src_var.file, line = cell.location.start_line)
        
        if src_var.category == PE.EntKind.Function:
            use_stmt = file_line_stmt_dict[location].snippet
            direct_name = dest_var.qualified_name.split('.')[-1]
            if is_compose_change_stmt(use_stmt, direct_name):
                append_dep_dict.setdefault(dest_var.qualified_name, set()).add(src_var.qualified_name)
            else:
                src_snippet_id = EG.Snippet_Id(defined_name=src_var.qualified_name,
                                            location= EG.Stmt_Location(
                                            file = src_var.file,
                                            line = src_var.location.start_line
                                            ))
                dependency_dict.setdefault(src_snippet_id, set()).add((used_name, location))    
        elif src_var.category in PE.Scope_EntKinds:
            location_use_nodes.setdefault(location, set()).add((used_name, location))
        else:
            raise ValueError(f"Unexpected src category {src_var.category} in {cell.src} to {cell.dest}")
    
    supper_snippet_dict = clear_unnecessary_supper_snippet_dict(supper_snippet_dict,id_raw_dict, pa.cells)
    snippet_EN_dep = construct_dependency(dependency_dict, location_use_nodes,node_def_locations) 
    resolve_inheritance(base_sub_dict, class_attr_dict, attr_class_dict)
    Entity_Graph = construct_graph(project, snippet_EN_dep, node_def_locations, file_line_stmt_dict, supper_snippet_dict, def_name_dict,append_dep_dict,supper_cls_dict)
    Entity_Graph.class_attr_dict = class_attr_dict
    Entity_Graph.attr_class_dict = attr_class_dict
    return Entity_Graph

def generate_graph_for_project(project: str):
    PA_Json_File = Raw_PyAnalyzer / f"{project}.json"
    pa = PM.JLoad_PyAnalyzerResult(PA_Json_File)
    Entity_Graph = from_pyanalyzer(pa, project)
    output_path = EntityGraph_Path / f"{project}.json"
    silent_Write(Entity_Graph.model_dump_json(indent=4), output_path)
        
if __name__ == "__main__":
    for project in projects:
        print(f"Processing {project}...")
        generate_graph_for_project(project)