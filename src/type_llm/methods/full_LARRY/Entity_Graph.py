from pydantic import BaseModel, Field, field_validator
from typing import Optional,Set,Dict,Union, Tuple
from enum import Enum
from networkx import DiGraph, strongly_connected_components, topological_sort,spring_layout
import networkx as nx
import ast
from type_llm.utils.log_manager import setup_logger
import json
from pathlib import Path
from typing import List,Iterable
from type_llm.utils.LLM import concurrent_llm,concurrent_conversation, concurrent_llm_check_dep, concurrent_dep_fix
from type_llm.utils.template import *
from type_llm.utils.config import EntityGraph_Path, Validation_Path,prototype_Path, Scope_Path, projects, LLM_Result_Path, illustrate_path, prototype_Path
import re
from type_llm.methods.full_LARRY.Validation import run_mypy,process_project,extract_msg_error_detail, extract_comment, restore_comments
import os
from type_llm.methods.full_LARRY.Dependency_Fixer import Stmt_Location, Snippet_Id
from type_llm.utils import silent_Write
import numpy as np
import os
import logging
import sys
import shutil
import random
from copy import deepcopy

def _get_posonlyargs(args_node: ast.arguments) -> list:
    return getattr(args_node, 'posonlyargs', [])

def _compare_arg_list(list1: list, list2: list) -> bool:
    if len(list1) != len(list2):
        return False
    return all(a.arg == b.arg for a, b in zip(list1, list2))

def _compare_vararg(var1: ast.arg, var2: ast.arg) -> bool:
    if not var1 and not var2:
        return True
    if not var1 or not var2:
        return False
    return var1.arg == var2.arg

def _compare_kwarg(kw1: ast.arg, kw2: ast.arg) -> bool:
    if not kw1 and not kw2:
        return True
    if not kw1 or not kw2:
        return False
    return kw1.arg == kw2.arg

def same_arg_list(args1, args2):
    if not _compare_arg_list(_get_posonlyargs(args1), _get_posonlyargs(args2)):
        return False
    if not _compare_arg_list(args1.args, args2.args):
        return False

    # 比较*args的存在性和名称
    if not _compare_vararg(args1.vararg, args2.vararg):
        return False

    # 比较仅关键字参数（名称和顺序）
    if not _compare_arg_list(args1.kwonlyargs, args2.kwonlyargs):
        return False

    # 比较**kwargs的存在性和名称
    if not _compare_kwarg(args1.kwarg, args2.kwarg):
        return False

    # 验证默认值参数的位置（不比较值内容）
    if len(args1.defaults) != len(args2.defaults):
        return False
    if len(args1.kw_defaults) != len(args2.kw_defaults):
        return False

    return True


def draw_graph(G:DiGraph, down_map, file_name:str):
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Bbox

    def adjust_labels(pos, labels, ax, font_size=12, max_iterations=50):
        text_objects = {}
        
        # 创建初始标签对象
        for node, (x, y) in pos.items():
            text = ax.text(x, y, str(labels[node]), 
                        fontsize=font_size, 
                        ha='center', va='center',
                        fontweight='bold')
            text_objects[node] = text
        
        ax.figure.canvas.draw()  # 确保bbox计算正确
        
        # 迭代优化标签位置
        for _ in range(max_iterations):
            moved = False
            objects = list(text_objects.items())
            
            for i, (node1, text1) in enumerate(objects):
                bbox1 = text1.get_window_extent().transformed(ax.transData.inverted())
                
                for j in range(i+1, len(objects)):
                    node2, text2 = objects[j]
                    bbox2 = text2.get_window_extent().transformed(ax.transData.inverted())
                    
                    # 使用overlaps()方法检查重叠
                    if bbox1.overlaps(bbox2):
                        moved = True
                        # 计算节点1到节点2的向量
                        dx = pos[node2][0] - pos[node1][0]
                        dy = pos[node2][1] - pos[node1][1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        
                        if dist < 1e-5:  # 避免除以零
                            dx, dy = 0.01, 0
                            dist = 0.01
                        
                        # 根据节点位置确定排斥方向
                        sign_x = np.sign(dx) if abs(dx) > 0.05 else 1
                        sign_y = np.sign(dy) if abs(dy) > 0.05 else 1
                        
                        # 只在不与原有位置有重大偏移的情况下移动标签
                        displacement = min(0.05, max(0.005, 0.1/(dist+1e-5)))
                        
                        # 为两个标签施加反向的力
                        text1.set_position((pos[node1][0] - displacement*sign_x, 
                                        pos[node1][1] - displacement*sign_y))
                        text2.set_position((pos[node2][0] + displacement*sign_x, 
                                        pos[node2][1] + displacement*sign_y))
            
            if not moved:
                break  # 如果没有标签移动，提前结束迭代
        
        return text_objects
    
    plt.figure(figsize=(10, 10))
    positions = [(1,7),(1,5),(5,9),(5,7),(1,9),(1,3),(5,3),(5,1), (9,9), (9,1),(1,1)]
    pos = {
        node: (x, y) for node, (x, y) in zip(G.nodes, positions)
    }
    node_list = list(G.nodes)
    node_color = ["#66c2a5" if down_map[node] else "#fc8d62" for node in node_list]
    nx.draw_networkx_nodes(G, pos, nodelist = node_list, node_size=600, node_color=node_color, alpha=0.9)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=30, 
                        width=3.0, edge_color="#7e7e7e", 
                        connectionstyle='arc3,rad=0.1')
    ax = plt.gca()
    node_labels = {n: str(n).replace("example.example.","") for n in G.nodes()}
    
    adjust_labels(pos, node_labels, ax, font_size=20)

    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()

logger = setup_logger()
logger.addHandler(logging.StreamHandler(stream=sys.stdout))
max_conversation  = 9

def ast_length(astNode):
    return ast.unparse(astNode).count('\n') + 1

def remove_redundant_imports(msg_list, environment_stmts):
          
    fault_symbol_set = set()
    
    for msg in msg_list:
        if 'already defined' in msg['description']:
            fault_symbol_set.add(re.findall(r'Name "(\w+)"', msg['description'])[0])
    logger.debug(f"found fault symbol: {fault_symbol_set}")

    new_environment_stmts = []
    for node in environment_stmts:
        if isinstance(node, (ast.Import,ast.ImportFrom)):
            new_name_list = []
            for alias in node.names:
                if alias.asname:
                    checked_name = alias.asname
                else:
                    checked_name = alias.name
                if checked_name not in fault_symbol_set:
                    new_name_list.append(alias)
                else:
                    logger.info(f"remove {alias.name}")
            if len(new_name_list) > 0:
                node.names = new_name_list
                new_environment_stmts.append(node)
        elif isinstance(node, ast.Assign):
            new_target_list = []
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id not in fault_symbol_set:
                    new_target_list.append(target)
            if len(new_target_list) > 0:
                node.targets = new_target_list
                new_environment_stmts.append(node)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id not in fault_symbol_set:
                new_environment_stmts.append(node)
        elif isinstance(node, (ast.FunctionDef,ast.AsyncFunctionDef)):
            if node.name not in fault_symbol_set:
                new_environment_stmts.append(node)
        else:
            new_environment_stmts.append(node)
    logger.debug(f"new environment stmts: {[ast.unparse(stmt) for stmt in new_environment_stmts]}")
    return new_environment_stmts
    # for file,fault_symbol_set in fault_symbol_set.items():
    #     full_path = os.getcwd() + '/' + file
    #     ast_tree = ast.parse(open(full_path).read())
    #     import_killer = Import_Killer(fault_symbol_set)
    #     ast_tree = import_killer.visit(ast_tree)  
    #     ast_tree = ast.fix_missing_locations(ast_tree)
    #     with open(full_path, 'w') as f:
    #         f.write(ast.unparse(ast_tree))
    # msg_list = run_mypy(Validation_Path / project)
    # return msg_list

def find_imports(file_name):
    class ImportFinder(ast.NodeVisitor):
        def __init__(self):
            self.imports = set()

        def visit_Import(self, node):
            self.imports.add(ast.unparse(node))
        def visit_ImportFrom(self, node):
            self.imports.add(ast.unparse(node))
    
    full_name = prototype_Path / file_name
    if not full_name.exists():
        return set()
    with open(full_name, 'r') as f:
        tree = ast.parse(f.read())
        finder = ImportFinder()
        finder.visit(tree)
        return finder.imports


def split_graph(g: nx.DiGraph, max_nodes: int) -> int:
    """
    将一个强连通的有向图 g 通过删除尽可能少的边拆分，
    使得拆分后所有强连通分量的节点数均不超过 max_nodes。
    
    算法思路：不断检查当前图中的强连通分量，挑选出结点数大于 max_nodes 的那部分，
    对该部分图（诱导子图）中按固定顺序遍历所有边，
    尝试删除一条边，看是否可以使该部分拆分成至少两个强连通分量，
    一旦成功就在原图中永久删除该边，并计数；如果没有边能使拆分发生，则删除固定（第一条）边，
    重复此过程直到所有强连通分量均满足节点数要求。
    
    参数：
      g: networkx.DiGraph 类型的强连通图
      max_nodes: 每个子连通分量最大的节点数量
      
    返回：
      被删除的边数（int）
    """

    # 为保证可重复性（虽然本代码中主要依赖排序保证确定性），可以设置随机种子
    random.seed(42)

    removed_edges = []  # 记录被删除的边
    # 在副本上操作，避免破坏原图
    # H = g.copy()
    H = g

    # 主循环：只要还有强连通分量中节点过多就继续
    while True:
        # 得到当前所有强连通分量
        scc_list = list(nx.strongly_connected_components(H))
        # 找出其中节点数大于 max_nodes 的分量（注意：若 g 本身刚好符合条件，则不做任何处理）
        too_big = [comp for comp in scc_list if len(comp) > max_nodes]
        if not too_big:
            break

        # 为了有确定性，选择最大的那个分量（也可以改为按固定序列选择）
        comp = max(too_big, key=lambda c: len(c))
        # 在 H 中取出该分量的诱导子图
        H_comp = H.subgraph(comp).copy()

        # 为了可重复性，对边按字典序排序（边为 (u, v) 元组）
        candidate_edges = list(H_comp.edges())
        candidate_edges.sort()

        edge_deleted = False
        # 尝试依次检查每条边删除是否能使 H_comp 拆分成多个 scc
        for edge in candidate_edges:
            # 复制一份测试用的子图
            H_test = H_comp.copy()
            H_test.remove_edge(*edge)
            new_scc = list(nx.strongly_connected_components(H_test))
            if len(new_scc) > 1:
                # 如果拆分成多个部分，则在原图 H 中删除该边，并退出循环
                H.remove_edge(*edge)
                removed_edges.append(tuple(edge))
                edge_deleted = True
                break

        if not edge_deleted:
            # 如果在该大分量中没有任何边一删除就能使其变为多个 scc，
            # 那么我们直接删除排在第一位的边。这样可能不会立即使得该 scc 拆分，
            # 但随着后续重复删除最终能拆分（算法是贪心启发式，不能保证最优，但符合“尽可能少删除”的要求）。
            edge = candidate_edges[0]
            H.remove_edge(*edge)
            removed_edges.append(tuple(edge))

    return removed_edges

def contract_scc(graph):
    # 查找所有强连通分量
    sccs = list(strongly_connected_components(graph))
    scc_length = [len(scc) for scc in sccs]
    remove_edges = []
    if len(scc_length) > 0 and max(scc_length) > 5:
        remove_edges = split_graph(graph, 5)

        print(f"Removed {remove_edges} edges to split SCC of size {max(scc_length)} into multiple components")
        sccs = list(strongly_connected_components(graph))
        scc_length = [len(scc) for scc in sccs]
        print("")
    
    # 创建节点到代表节点的映射
    node_to_contracted = {}
    for scc in sccs:
        representative = tuple(scc)  # 假设节点可比较，取最小节点作为代表
        for node in scc:
            node_to_contracted[node] = representative
    
    # 创建新图
    contracted_graph = DiGraph()
    
    # 添加收缩后的节点（自动去重）
    contracted_graph.add_nodes_from(set(node_to_contracted.values()))
    
    # 收集跨分量的边
    edges = set()
    for u, v in graph.edges():
        u_rep = node_to_contracted[u]
        v_rep = node_to_contracted[v]
        if u_rep != v_rep:
            edges.add((u_rep, v_rep))
    
    # 添加边到新图
    contracted_graph.add_edges_from(edges)
    
    # 根据收缩后的边计算拓扑序
    return  list(topological_sort(contracted_graph)), scc_length, remove_edges

def add_annotation(snippet:str, comment_dict:dict):    
    new_lines = []
    for i, line in enumerate(snippet.split('\n')):
        if i in comment_dict:
            new_lines.append(f"{line} # {', '.join(comment_dict[i])}")
        else:
            new_lines.append(line)
    annotated_snippet ='\n'.join(new_lines)
    logger.info(annotated_snippet)
    return annotated_snippet

def collect_type_ann(code_snippet:str,allowed_names:Set[str]):
    collected_dict = {}
    for i, line in enumerate(code_snippet.split('\n')):
        comment = line.split('#')[1].strip() if '#' in line else ''
        if comment:
            deps = [i.strip() for i in comment.split(',')]
            deps = [i for i in deps if i in allowed_names]
            if deps:
                collected_dict[i] = set(deps)
    return collected_dict

class Entity_Type(str, Enum):
    var = "var"
    func = "func"
    class_ = "class"

class Entity_Class(str, Enum):
    Function = "func"
    Class = "cls"
    Variable = "var"
    Module = "module"

class Scope(BaseModel, frozen=True):
    file: str
    module_scope: Optional[str]
    class_scope: Optional[str]
    function_scope: Optional[Tuple[str, int]]
    current_scope: str
    scope_class: Entity_Class
    global_set: Set[str]
    
    def __str__(self):
        return json.dumps({
            "file": self.file,
            "module_scope": self.module_scope,
            "class_scope": self.class_scope,
            "current_scope": self.current_scope,
            "scope_class": self.scope_class,
            "global_set": str(self.global_set)
        }, indent=4)

class Snippet(BaseModel, frozen=True):
    snippet: str
    scope: Scope

class Snippet_Collector(ast.NodeVisitor):
    def __init__(self, file:str, module:str, line_dict:Dict[int, Optional[Snippet]]):
        self.line_dict = line_dict
        self.scope = Scope(file=file, 
                           current_scope=module, 
                           module_scope=module,
                           class_scope=None,
                           function_scope = None,
                           scope_class=Entity_Class.Module,
                           global_set = set())
        
    def generic_visit(self, node):
        if isinstance(node, (ast.stmt, ast.expr)) and node.lineno in self.line_dict and self.line_dict[node.lineno] is None:
            self.line_dict[node.lineno] = Snippet(
                                            snippet = ast.unparse(node),
                                            scope = self.scope
                                            )
        super().generic_visit(node)
    
    def visit_FunctionDef(self, node):
        global_set = set()
        for stmt in node.body:
            if isinstance(stmt, ast.Global):
                global_set.update(stmt.names)
        old_scope = self.scope
        self.scope = Scope(file=old_scope.file, 
                           current_scope=old_scope.current_scope+'.'+node.name, 
                           module_scope=old_scope.module_scope,
                           class_scope=old_scope.class_scope,
                           function_scope=(old_scope.current_scope+'.'+node.name, node.lineno),
                           scope_class=Entity_Class.Function,
                           global_set = global_set)    
        self.generic_visit(node)
        self.scope = old_scope
    
    def visit_ClassDef(self, node):
        old_scope = self.scope
        self.scope = Scope(file=old_scope.file, 
                           current_scope=old_scope.current_scope+'.'+node.name, 
                           module_scope=old_scope.module_scope,
                           class_scope=old_scope.current_scope+'.'+node.name,
                           function_scope=old_scope.function_scope,
                           scope_class=Entity_Class.Class,
                           global_set = set())    
        self.generic_visit(node)
        self.scope = old_scope
        
    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

def extract_snippet_from_dir(dir:Path, module:str, file_line_dict:Dict[str, Dict[int, Optional[Snippet]]]):
    for file in os.listdir(dir):
        current_path = dir / file
        if file.endswith(".py") and os.path.isfile(current_path):
            relative_path = str(current_path.relative_to(prototype_Path))
            if relative_path in file_line_dict:
                current_module = module + '.' + file[:-3]
                with open(current_path, "r") as f:
                    astNode = ast.parse(f.read())
                    sc = Snippet_Collector(file=relative_path, 
                                           module=current_module,
                                           line_dict = file_line_dict[relative_path])
                    sc.visit(astNode)
                    file_line_dict[relative_path] = sc.line_dict
        elif os.path.isdir(current_path):
            extract_snippet_from_dir(current_path, module + '.' + file, file_line_dict)

def get_defined_names(target,scope:Scope):
    if isinstance(target, ast.Name):
        if scope.scope_class == Entity_Class.Function:
            if target.id not in scope.global_set:
                return scope.current_scope+'.'+target.id, 1
            else:
                return scope.module_scope+'.'+target.id, 0
        else:
            return scope.current_scope+'.'+target.id, 0 
        
    elif isinstance(target, ast.Attribute) \
        and isinstance(target.value, ast.Name) \
            and target.value.id in ('self','cls') \
                and scope.scope_class == Entity_Class.Function \
                    and scope.class_scope is not None:
        return scope.class_scope+'.'+target.attr, 0
    else:
        return None, 2

class Snippet_Unit(BaseModel):
    ID: Snippet_Id
    snippet: Optional[str] = None
    dependency: Optional[Set[Tuple[str, str]]] = Field(default_factory=set)
    lacked_symbols: Set[str]= Field(default_factory=set)
    possible_types: Dict[str, set[str]] = Field(default_factory=dict)
    supper_snippet: Optional[Snippet_Id] = None
    
    def __hash__(self):
        return hash((self.snippet, frozenset(self.dependency), self.supper_snippet))
    
    def extract_used_attrs(self):
        class Attr_Searcher(ast.NodeVisitor):
            def __init__(self):
                self.attr_set = set()
            def visit_Attribute(self, node:ast.Attribute):
                self.attr_set.add(node.attr) 
        attr_searcher = Attr_Searcher()
        astNode = ast.parse(self.snippet)
        attr_searcher.visit(astNode)
        return attr_searcher.attr_set        



class Entity_Node(BaseModel):
    qualified_name: str 
    category: Entity_Type
    def_file: str
    def_name: str
    stub_code: Optional[str] = None
    dependency: Optional[Set[str]] = Field(default_factory=set)
    snippet_set: Optional[Set[Snippet_Unit]] = Field(default_factory=set)
    appended_dependency: Optional[Set[str]] = Field(default_factory=set)
    all_objs_in_def_file: Optional[str] = None
    
    def append_possible_type(self, class_attr_dict:Dict[str, Dict[str,Set[str]]], attr_class_dict:Dict[str, Set[str]]):
        if self.category != Entity_Type.func:
            logger.warning(f"append_possible_type only support func, but get {self}")
            return
        
        for snippet in self.snippet_set:
            if snippet.possible_types:
                # logger.error(f"snippet {snippet} already has possible_types")
                continue
            attr_set = snippet.extract_used_attrs()
            if attr_set:
                pass
            possible_type_set = set()
            for used_attr in attr_set:
                possible_type_set.update(attr_class_dict.get(used_attr, set()))   
            
            possible_type_info = {
                possible_type: set(class_attr_dict.get(possible_type, {}).keys()) & set(attr_set)
                for possible_type in possible_type_set
            }
            snippet.possible_types = possible_type_info
    
    def generate_attr_dependency(self, class_attr_dict:Dict[str, Dict[str,Set[str]]]):        
        if self.category != Entity_Type.class_:
            logger.warning(f"generate_attr_dependency only support class, but get {self}")
            return
        if len(self.snippet_set) > 1:
            raise Exception(f"Class Node {self} has more than one snippet")
        all_attr_set = set()
        for attr_set in class_attr_dict.get(self.qualified_name,{}).values():
            all_attr_set.update(attr_set)
        
        self.stub_code = """
        class {class_name}:
            attrs:
                {attr_str}
        """.format(class_name=self.qualified_name, attr_str="\n".join([f"{attr}" for attr in all_attr_set]))
        self.dependency = set()#all_attr_set
           
class AnalyzeUnit(BaseModel):
    unit_cluster: Set[str]
    dependency: Set[str]

def clear_func_ann(node, Entity_node:Entity_Node):
    class Func_Cleaner(ast.NodeTransformer):
        def __init__(self, base_module, full_name):
            self.target_name = full_name
            self.base_module = base_module
            self.need_any = False
        def visit_ClassDef(self, node):
            history_name = self.base_module
            self.base_module = self.base_module + '.' + node.name
            new_node = super().generic_visit(node)
            self.base_module = history_name
            return new_node
           
        def visit_FunctionDef(self, node):
            current_name = self.base_module+'.'+node.name
            if current_name == self.target_name:
                return generate_pure_func_nodes(node)
            return node
        
        def visit_AsyncFunctionDef(self, node):
            return self.visit_FunctionDef(node)
        
        def visit_AnnAssign(self, node):
            if isinstance(node.target, ast.Name) and self.base_module+'.'+node.target.id == self.target_name:
                node.annotation = ast.Name(id="Any", ctx=ast.Load())
                self.need_any = True
            return node
    
    base_module = '.'.join( Entity_node.def_file[:-3].split('/') )
    full_name = Entity_node.qualified_name
    fc = Func_Cleaner(base_module, full_name)
    retNode = fc.visit(node)
    need_any = fc.need_any
    return retNode, need_any        
     

def get_cluster_id(cluster:Set[str]):
    return repr(sorted(list(cluster)))

def extract_names(code:str):
    class Name_Collector(ast.NodeVisitor):
        def __init__(self):
            self.names = set()
            self.local_names = set()
        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load):
                self.names.add(node.id)
            elif isinstance(node.ctx, ast.Store):
                self.local_names.add(node.id)
                
        def visit_FunctionDef(self, node):
            if hasattr(node.args, 'posonlyargs'):
                for arg in node.args.posonlyargs:  # Python 3.8+ 仅位置参数
                    self.local_names.add(arg.arg)
            
            for arg in node.args.args:  # 普通位置参数
                self.local_names.add(arg.arg)
            
            # 2. 可变参数 *args
            if node.args.vararg:
                self.local_names.add(node.args.vararg.arg)
            
            # 3. 关键字限定参数
            for arg in node.args.kwonlyargs:
                self.local_names.add(arg.arg)
            
            # 4. 关键字参数 **kwargs
            if node.args.kwarg:
                self.local_names.add(node.args.kwarg.arg)
            return super().generic_visit(node)            
        
    astNode = ast.parse(code)
    collector = Name_Collector()
    collector.visit(astNode)
    return collector.names, collector.local_names

def generate_pure_func_nodes(def_node, default_ann = None):
    new_args = ast.arguments(
        args=[ast.arg(arg=origin_arg.arg, annotation= default_ann) for origin_arg in def_node.args.args ],
        posonlyargs=[ast.arg(arg=origin_arg.arg, annotation= default_ann) for origin_arg in def_node.args.posonlyargs],
        kwonlyargs=[ast.arg(arg=origin_arg.arg, annotation= default_ann) for origin_arg in def_node.args.kwonlyargs],
        kw_defaults=def_node.args.kw_defaults,
        defaults=def_node.args.defaults
    )
    if def_node.args.vararg:
        new_args.vararg = ast.arg(arg=def_node.args.vararg.arg, annotation= default_ann)
    if def_node.args.kwarg:
        new_args.kwarg = ast.arg(arg=def_node.args.kwarg.arg, annotation= default_ann)
    if isinstance(def_node, ast.FunctionDef):
        newNode = ast.FunctionDef(
            name = def_node.name, 
            args = new_args, 
            body = def_node.body,
            decorator_list=def_node.decorator_list)
    elif isinstance(def_node, ast.AsyncFunctionDef):
        newNode = ast.AsyncFunctionDef(
            name = def_node.name, 
            args = new_args, 
            body = def_node.body,
            decorator_list=def_node.decorator_list)
    else:
        raise Exception(f"def_node {def_node} is not a FunctionDef or AsyncFunctionDef")
    if def_node.returns:
        newNode.returns = None
    newNode = ast.fix_missing_locations(newNode)
    return newNode
          
def extract_anno_dict(def_node, default_ann = None):
    anno_dict = {}
    anno_dict['posonlyargs'] = [arg.annotation if arg.annotation else default_ann for arg in def_node.args.posonlyargs]
    anno_dict['args'] = [arg.annotation if arg.annotation else default_ann for arg in def_node.args.args]
    anno_dict['kwonlyargs'] = [arg.annotation if arg.annotation else default_ann for arg in def_node.args.kwonlyargs]
    anno_dict['vararg'] = def_node.args.vararg.annotation if def_node.args.vararg else default_ann 
    anno_dict['kwarg'] = def_node.args.kwarg.annotation if def_node.args.kwarg else default_ann
    anno_dict['returns'] = def_node.returns
    return anno_dict
    
class Entity_Graph(BaseModel, arbitrary_types_allowed=True):
    project:str
    snippet_dict : Dict[Snippet_Id, Snippet_Unit] = Field(default_factory=dict)
    node_dict : dict[str, Entity_Node] = Field(default_factory=dict)
    class_attr_dict:Dict[str, Dict[str,Set[str]]] = Field(default_factory=dict)
    attr_class_dict:Dict[str, Set[str]] = Field(default_factory=dict)
    file_import_dict:Dict[str, Set[str]] = Field(default_factory=dict)
    supper_cls_dict:Dict[str, Set[str]] = Field(default_factory=dict)
    removed_edges:List[Tuple[str, str]] = Field(default_factory=list)
    comment_dict:Dict[str, Dict[int,str]] = Field(default_factory=dict)
    # node_graph: Optional[DiGraph] = None
    
    @field_validator('snippet_dict', mode='before')
    def check_snippet_dict(cls, v):
        new_dic = dict()
        for snippet_id, snippet in v.items():
            if isinstance(snippet_id, str):
                new_snippet_id = snippet_id.replace("' location=Stmt_Location", "', location=Stmt_Location")
                new_dic[eval(f"Snippet_Id({new_snippet_id})", globals(), locals())] = snippet
            else:
                new_dic[snippet_id] = snippet
        return new_dic
                     
    def preprocess(self):
        #preprocess
        #整理snippet的实体依赖
        valid_dep = set(self.node_dict.keys())
        #整理node的依赖
        for nodeName, node in self.node_dict.items():
            #实体依赖
            for snippet in node.snippet_set:
                snippet.snippet = snippet.snippet.replace('```','')
                snippet.dependency = set([dep for dep in snippet.dependency if dep[0] in valid_dep])
            node.dependency.intersection_update(valid_dep)
            node.dependency.update(node.appended_dependency)
            #其他外部体依赖
            if node.category == Entity_Type.func:
                node.append_possible_type(self.class_attr_dict, self.attr_class_dict)
            elif node.category == Entity_Type.class_:
                node.generate_attr_dependency(self.class_attr_dict)
            with open(Validation_Path / (node.def_file) ) as f:
                lines = f.readlines()
            for l in lines:
                if "__all__" in l and "= [" in l:
                    try:
                        node.all_objs_in_def_file = re.search(r"(?<=\=\s)\[.*\]", l).group(0)
                    except:
                        print(node.def_file,'\n', l)
                        exit()
                    break
    
    def analyze(self, iteration):
        #generate graph
        analyze_graph = DiGraph()
        for node in self.node_dict.values():
            #图中只保留未生成stub的节点
            if node.stub_code is None:
                analyze_graph.add_node(node.qualified_name)
            removed_dependency = []
            for dependency in node.dependency:
                if dependency in self.node_dict:
                    #图中只保留未生成stub的节点
                    if self.node_dict[dependency].stub_code is None:
                        analyze_graph.add_edge(dependency, node.qualified_name)
                else:
                    logger.warning(f"dependency {dependency} not found in node_dict")
                    removed_dependency.append(dependency)
            node.dependency = set([dep for dep in node.dependency if dep not in removed_dependency])
        
        # to_show = DiGraph()
        # for node in self.node_dict.values():
        #     #图中只保留未生成stub的节点
        #     to_show.add_node(node.qualified_name)
        #     for dependency in node.dependency:
        #         if dependency in self.node_dict:
        #                 to_show.add_edge(dependency, node.qualified_name)
                
        # down_map = {node_name: node.stub_code is None for node_name, node in self.node_dict.items()}
        # draw_graph(to_show, down_map ,illustrate_path/f"analyze_graph_{iteration}.png")
        #收缩强连通分量，获取拓扑序
        analyze_sequence, scc_length,removed_edges = contract_scc(analyze_graph)
        for dep, node in removed_edges:
            if dep in self.node_dict[node].dependency:
                self.node_dict[node].dependency.remove(dep)
            if dep in self.node_dict[node].appended_dependency:
                self.node_dict[node].appended_dependency.remove(dep)
        self.removed_edges.extend(removed_edges)
        logger.info(f"scc_length: {scc_length}")
        for scc in analyze_sequence:
            if len(scc) > 1:
                logger.warning(f"analyze_sequence has scc: {scc}")
        #根据拓扑序进行分析
        self.type_inference(analyze_sequence, iteration)
    
    def type_inference(self,
                    analyze_sequence:List[Iterable[str]],
                    iteration:int):
        
        # 初始化 未处理cluster列表 和 已处理节点映射
        waiting_list:List[AnalyzeUnit] = []
        known_stub = {}
        for node_name in self.node_dict:
            if self.node_dict[node_name].stub_code is not None:
                known_stub[node_name] = self.node_dict[node_name].stub_code
        for unit_cluster in analyze_sequence:
            #对每个强连通分量，统计其依赖
            if not unit_cluster:
                continue
            dependent_ids = set()
            new_cluster = set()
            for node_name in unit_cluster:
                node = self.node_dict[node_name]
                if node.stub_code is None:
                    dependent_ids.update(node.dependency)
                    new_cluster.add(node_name)
            if new_cluster:
                waiting_list.append(AnalyzeUnit(unit_cluster=new_cluster, dependency=dependent_ids-new_cluster))
        
        #迭代分析，直到所有节点都被处理
        logger.info(f"已处理节点数: {len(known_stub)}, 未处理节点数: {len(waiting_list)}")
        #收集下一批LLM分析的cluster,并更新"未处理cluster"的依赖
        next_batch:List[Set[str]] = []
        updated_waiting_list = []
        waiting_dep = set()
        for au in waiting_list:
            au.dependency = au.dependency - set(known_stub.keys()) 
            if not au.dependency:
                next_batch.append(au.unit_cluster)
            else:
                waiting_dep.update(au.dependency)
                updated_waiting_list.append(au)
        waiting_list = updated_waiting_list
        #调用LLM
        LLM_Result = self.call_LLM_check(next_batch, iteration)
        #更新dependency
        valid_batch = []
        new_LLM_Conv = {}
        for cluster in next_batch:
            cluster_id = get_cluster_id(cluster)
            if cluster_id not in LLM_Result:
                continue
            LLM_conversation = LLM_Result[cluster_id]
            if LLM_conversation is None:
                dep_list,lack_list = [], []
            else:
                known_deps = set()
                for node_name in cluster:
                    known_deps.update(self.node_dict[node_name].dependency)
                dep_list,lack_list = self.extract_dependency(cluster, LLM_conversation,known_deps)
            if lack_list:
                LLM_conversation.append(
                    {
                        "role": "user",
                        "content":reAnalyze_Dep_template.format(project=self.project, wrong_name_list=lack_list)
                    }
                )
                new_LLM_Conv[cluster_id] = LLM_conversation
            else:
                print("pass")
                Updated = False
                for node_name in eval(cluster_id):
                    for dep in dep_list:
                        if (dep,node_name) not in self.removed_edges:
                            self.node_dict[node_name].appended_dependency.add(dep)
                            logger.info(f" append dependency {dep}->{node_name}")
                            Updated = True    
                if not Updated:
                    valid_batch.append(cluster)           
        
        idx = 0
        MAX_REPAIR_ROUND = 4
        while new_LLM_Conv and idx < MAX_REPAIR_ROUND:
            logger.info(f"Resolve Dependency Errors: the {idx} / {MAX_REPAIR_ROUND} round, {len(new_LLM_Conv)} errors")
            saved_path = LLM_Result_Path / f"{self.project}" / f"Dependency_Fix_{iteration}_{idx}.json"
            
            if saved_path.exists():
                with open(saved_path, 'r') as f:
                    LLM_Result = json.load(f)
                    LLM_Result = {key: value for key, value in LLM_Result.items() if value is not None}
            else:
                LLM_Result = {}

            concurrent_dep_fix(new_LLM_Conv, LLM_Result, saved_path)
            idx += 1
            next_Conv_batch:Dict = {}
            for cluster_id in new_LLM_Conv:
                if not eval(cluster_id):
                    continue
                if cluster_id not in LLM_Result:
                    continue
                LLM_conversation = LLM_Result[cluster_id]
                cluster = eval(cluster_id)
                if LLM_conversation is None:
                    dep_list,lack_list = [],[]
                else:
                    known_deps = set()
                    for node_name in cluster:
                        known_deps.update(self.node_dict[node_name].dependency)
                    dep_list,lack_list = self.extract_dependency(cluster, LLM_conversation, known_deps)
                if lack_list:
                    LLM_conversation.append(
                    {
                        "role": "user",
                        "content":reAnalyze_Dep_template.format(project=self.project, wrong_name_list=lack_list)
                    }
                )
                    next_Conv_batch[cluster_id] = LLM_conversation
                else:
                    print("pass")
                    Updated = False
                    for node_name in eval(cluster_id):
                        for dep in dep_list:
                            if (dep,node_name) not in self.removed_edges:
                                self.node_dict[node_name].appended_dependency.add(dep)
                                logger.info(f" append dependency {dep}->{node_name}")
                                Updated = True    
                    if not Updated:
                        valid_batch.append(cluster)
                                    
            new_LLM_Conv = next_Conv_batch
        if new_LLM_Conv:
            for cluster_id in new_LLM_Conv:
                if not eval(cluster_id):
                    continue
                if cluster_id not in LLM_Result:
                    continue
                cluster = eval(cluster_id)
                LLM_conversation = LLM_Result[cluster_id]
                if LLM_conversation is None:
                    dep_list, lack_list = [],[]
                else:
                    LLM_conversation = LLM_conversation[:-1] #去掉最后添加的未回答的提问
                    known_deps = set()
                    for node_name in cluster:
                        known_deps.update(self.node_dict[node_name].dependency)
                    dep_list,lack_list = self.extract_dependency(cluster, LLM_conversation,known_deps)
                Updated = False
                for node_name in eval(cluster_id):
                    for dep in dep_list:
                        if (dep,node_name) not in self.removed_edges:
                            self.node_dict[node_name].appended_dependency.add(dep)
                            logger.info(f" append dependency {dep}->{node_name}")
                            Updated = True    
                if not Updated:
                    valid_batch.append(cluster)
        
        logger.info("-------------Dependency Analysis Done---------------")
        logger.info(f"#Valid Clusters: {len(valid_batch)}")
        next_batch = valid_batch    
        LLM_Result = self.call_LLM(next_batch, iteration)
        #更新known_stub
        new_LLM_Conv = {}
        for i,cluster in enumerate(next_batch):
            if not cluster: #scc 聚合算法会产生空cluster，跳过
                continue
            # if 'pre_commit.envcontext.envcontext' not in cluster:
            #     continue
            # else:
            #     pass
            cluster_id = get_cluster_id(cluster)
            if cluster_id not in LLM_Result:
                logger.warning(f"Cluster {cluster_id} not found in LLM Result")
            LLM_conversation = LLM_Result[cluster_id]
            logger.info(f"processing cluster {i} / {len(next_batch)}")

            if LLM_conversation is None:
                for node_name in cluster:
                    logger.info(f"No response from LLM for {cluster_id}")
                    known_stub[node_name] = "Any"
                    self.node_dict[node_name].stub_code = "Any"
            else:
                stub_dict,import_stmts = self.generate_stub(cluster, LLM_conversation)
                lacked_names = []
                
                for node_name in cluster:
                    if node_name not in stub_dict:
                        lacked_names.append(f"the stub format of {node_name} is not correct")
                if lacked_names:
                    failed_cluster = (lacked_names, [])
                else:
                    failed_cluster = self.check_validation(stub_dict, import_stmts)
                error_msg, previous_wrong_names = failed_cluster
                
                for node_name in previous_wrong_names:
                    logger.info(f"found previous wrong node {node_name} from {cluster_id}")
                    self.node_dict[node_name].stub_code = None
                    known_stub[node_name] = None
                    
                if not error_msg:
                    logger.info("pass")
                    logger.info(f"Cluster {cluster_id} passed validation")
                    for node_name in cluster:
                        known_stub[node_name] = '\n'.join([ast.unparse(node) for node in import_stmts]) +'\n'+ '\n'.join(stub_dict[node_name])
                        self.node_dict[node_name].stub_code = '\n'.join([ast.unparse(node) for node in import_stmts]) +'\n'+ '\n'.join(stub_dict[node_name])
                else:
                    logger.info(f'fail_cluster:{error_msg} in cluster {cluster_id}')
                    LLM_conversation.append(
                        {
                            "role": "user",
                            "content": error_template.format(error_msg = error_msg, cluster_name=repr(cluster))
                        }
                    )
                    new_LLM_Conv[cluster_id] = LLM_conversation
        
        idx = 0
        MAX_REPAIR_ROUND = 4
        while new_LLM_Conv and idx < MAX_REPAIR_ROUND:
            logger.info(f"Resolve Mypy Errors: the {idx} / {MAX_REPAIR_ROUND} round, {len(new_LLM_Conv)} errors")
            saved_path = LLM_Result_Path / f"{self.project}" / f"TypeInference_{iteration}_{idx}.json"
            
            if saved_path.exists():
                with open(saved_path, 'r') as f:
                    LLM_Result = json.load(f)
                    LLM_Result = {key: value for key, value in LLM_Result.items() if value is not None}
            else:
                LLM_Result = {}
            concurrent_conversation(new_LLM_Conv, LLM_Result, saved_path)
            idx += 1
            next_Conv_batch:Dict = {}
            for cluster_id in new_LLM_Conv:
                if not eval(cluster_id):
                    continue
                if cluster_id not in LLM_Result:
                    continue
                LLM_conversation = LLM_Result[cluster_id]
                cluster = eval(cluster_id)
                if LLM_conversation is None:
                    logger.info(f"No response from LLM for {cluster_id}")
                    for node_name in cluster:
                        known_stub[node_name] = "Any"
                        self.node_dict[node_name].stub_code = "Any"
                else:
                    stub_dict,import_stmts = self.generate_stub(cluster,LLM_conversation)
                    lacked_names = []
                    for node_name in cluster:
                        if node_name not in stub_dict:
                            lacked_names.append(f"the stub format of {node_name} is not correct")
                    if lacked_names:
                        failed_cluster = (lacked_names, [])
                    else:
                        failed_cluster = self.check_validation(stub_dict, import_stmts)
                    error_msg, previous_wrong_names = failed_cluster
                    
                    for node_name in previous_wrong_names:
                        logger.info(f"found previous wrong node {node_name} from {cluster_id}")
                        self.node_dict[node_name].stub_code = None
                        known_stub[node_name] = None
                        
                    if not error_msg:
                        logger.info("pass")
                        logger.info(f"Cluster {cluster_id} passed validation")
                        for node_name in cluster:
                            known_stub[node_name] = '\n'.join([ast.unparse(node) for node in import_stmts]) +'\n'+ '\n'.join(stub_dict[node_name])
                            self.node_dict[node_name].stub_code = '\n'.join([ast.unparse(node) for node in import_stmts]) +'\n'+ '\n'.join(stub_dict[node_name])
                    else:
                        logger.info(f'fail_cluster:{error_msg} in cluster {cluster_id}')
                        LLM_conversation.append(
                            {
                                "role": "user",
                                "content": error_template.format(error_msg = error_msg, cluster_name=repr(cluster))
                            }
                        )
                        if len(LLM_conversation) > max_conversation:
                            keep_length = 1 if idx < MAX_REPAIR_ROUND else 3
                            logger.info(f"conversation length {len(LLM_conversation)} of cluster {cluster_id} exceeds max_conversation {max_conversation}")
                            LLM_conversation = LLM_conversation[:keep_length]
                        next_Conv_batch[cluster_id] = LLM_conversation
            new_LLM_Conv = next_Conv_batch
        
        if new_LLM_Conv:
            logger.info(f"Unresolved Mypy Errors: {len(new_LLM_Conv)} errors")
            for cluster_id in new_LLM_Conv:
                LLM_conversation = new_LLM_Conv[cluster_id]
                cluster = eval(cluster_id)
                stub_dict,import_stmts = self.generate_stub(cluster,LLM_conversation[:-1])
                node_dict = self.last_try_stub(stub_dict,import_stmts)
                for node_name,node_annotation in node_dict.items():
                    known_stub[node_name] = '\n'.join(node_annotation)
                    self.node_dict[node_name].stub_code = '\n'.join(node_annotation)
          
    def inner_check_validation(self, stub_dict:Dict[str, set[str]], import_stmts:List, previous_wrong_names:List[str]):

        class Ann_Adder(ast.NodeTransformer):
            def __init__(self, name_parts:List[str], stub:Union[ast.FunctionDef, ast.AsyncFunctionDef], has_multiple:bool=False):
                self.name_parts = name_parts[:-1]
                self.target_name = name_parts[-1]
                self.stub = stub
                self.stack = []
                self.found = False
                self.inIfExp = False #为了避免在不同分支中重复定义，设置一个标志位
                self.extracted_node = None
                self.has_multiple = has_multiple
                self.added_lines = []
                
            def visit_ClassDef(self, node):
                self.stack.append(node.name)
                new_body = []
                for stmt in node.body:
                    new_body.append(self.visit(stmt))
                if self.stack == self.name_parts and not self.found:
                    if isinstance(self.stub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        logger.error(f"{ast.unparse(self.stub)} is not found in {node.name}")
                    if ast.get_docstring(node):
                        new_body.insert(1, self.stub)
                        self.added_lines.append( (node.lineno+1, ast_length(self.stub)) )
                    else:
                        new_body.insert(0, self.stub)
                        self.added_lines.append( (node.lineno, ast_length(self.stub)) )
                    self.found = True
                self.stack.pop()
                node.body = new_body
                return node
            
            def visit_FunctionDef(self, node):
                if self.stack ==self.name_parts and node.name == self.target_name:
                    if self.has_multiple and (ast.unparse(node.decorator_list) != ast.unparse(self.stub.decorator_list)): #setter, property
                        return node
                    if 'overload' in ast.unparse(node.decorator_list) and not same_arg_list(node.args, self.stub.args):
                        return node
                    self.found = True
                    node.args = self.stub.args
                    if node.args.args and node.args.args[0].arg == 'self':
                        node.args.args[0].annotation = None                        
                    if node.name not in ['__init__','__init_subclass__']:
                        node.returns = self.stub.returns
                return node
            
            def visit_AsyncFunctionDef(self, node):
                if self.stack ==self.name_parts and node.name == self.target_name:
                    if self.has_multiple and (ast.unparse(node.decorator_list) != ast.unparse(self.stub.decorator_list)):
                        return node
                    if 'overload' in ast.unparse(node.decorator_list) and not same_arg_list(node.args, self.stub.args):
                        return node
                    self.found = True
                    if self.inIfExp:
                        self.extracted_node = deepcopy(node)
                        self.extracted_node.args = self.stub.args
                        if node.args.args and node.args.args[0].arg == 'self':
                            self.extracted_node.args.args[0].annotation = None                        
                        if node.name not in ['__init__','__init_subclass__']:
                            self.extracted_node.returns = self.stub.returns
                        return node
                    node.args = self.stub.args
                    if node.args.args and node.args.args[0].arg == 'self':
                        node.args.args[0].annotation = None                        
                    if node.name not in ['__init__','__init_subclass__']:
                        node.returns = self.stub.returns
                return node
            
            def visit_AnnAssign(self, node):
                if self.stack ==self.name_parts and isinstance(node.target, ast.Name) and node.target.id == self.target_name:
                    self.found = True
                    if self.inIfExp:
                        if isinstance(self.stub, ast.Assign):
                        # logger.info(self.name_parts, self.target_name, ast.unparse(self.stub))
                            self.extracted_node = self.stub
                        else:
                            self.extracted_node = deepcopy(node)
                            self.extracted_node.annotation = self.stub.annotation
                        return node
                    if isinstance(self.stub, ast.Assign):
                        # logger.info(self.name_parts, self.target_name, ast.unparse(self.stub))
                        return self.stub
                    node.annotation = self.stub.annotation
                return node
            
            def visit_Assign(self, node):
                if self.stack ==self.name_parts and \
                    len(node.targets) == 1 and \
                    isinstance(node.targets[0], ast.Name) and \
                    node.targets[0].id == self.target_name:
                    self.found = True
                    if self.inIfExp:
                        if isinstance(self.stub, ast.AnnAssign):
                        # logger.info(self.name_parts, self.target_name, ast.unparse(self.stub))
                            self.extracted_node = self.stub
                        else:
                            self.extracted_node = deepcopy(node)
                            self.extracted_node.value = self.stub.value
                        return node
                    if isinstance(self.stub, ast.AnnAssign):
                        # logger.info(self.name_parts, self.target_name, ast.unparse(self.stub))
                        self.stub.value = node.value
                        return self.stub
                    else:
                        node.value = self.stub.value
                return node
            
            def visit_Module(self, node):
                new_body = []
                for stmt in node.body:
                    if self.found:
                        new_body.append(stmt)
                    else:
                        if isinstance(stmt, (ast.Assign,ast.AnnAssign,ast.FunctionDef, ast.ClassDef)):
                            new_body.append(self.visit(stmt))
                        elif isinstance(stmt, ast.If):
                            if not stmt.orelse or isinstance(self.stub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                new_body.append(self.visit(stmt))
                            elif isinstance(self.stub, (ast.Assign,ast.AnnAssign)) and self.stack == self.name_parts:
                                old_inIfExp = self.inIfExp
                                self.inIfExp = True
                                for in_stmt in stmt.body:
                                    self.visit(in_stmt)
                                if self.found:
                                    if not self.extracted_node:
                                        raise Exception(f"Could not extract node {self.name_parts},{self.target_name} from if statement: {ast.unparse(stmt)}")
                                    else:
                                        new_body.append(self.extracted_node)
                                        self.added_lines.append((stmt.lineno-1, ast_length(self.extracted_node)))
                                self.inIfExp = old_inIfExp
                                new_body.append(stmt)   
                            else:
                                new_body.append(stmt)
                        else:
                            new_body.append(stmt)
                node.body = new_body
                return node

        #remove wrong imports
        
        changed_dict = {}
        for node_name, stub_set in stub_dict.items():
            # if node_name == 'jinja2.filters._min_or_max':
            #     pass
            entity_node = self.node_dict[node_name]
            module_name = '.'.join(Path(entity_node.def_file[:-3]).parts)
            direct_name = node_name.replace(module_name+'.', '')
            name_parts = direct_name.split('.')
            file_path = Validation_Path / (entity_node.def_file)
            source_code = open(file_path).read()
            all_stubs = ''.join(stub_set)
            local_stmts = [deepcopy(import_stmt) for import_stmt in import_stmts if ast.unparse(import_stmt)+'\n' not in source_code and ast.unparse(import_stmt)+'  #type: ignore' not in source_code]
            offset = []
            new_imports = []
            all_names = entity_node.all_objs_in_def_file
            if all_names is None:
                all_names = '[]'
            for stmt in local_stmts:
                if isinstance(stmt, ast.ImportFrom):
                    if  stmt.module is not None and (module_name.endswith('.'+stmt.module) or module_name == stmt.module or 'ctypes' in stmt.module):
                        continue
                    else:
                        alias = []
                        for name in stmt.names:
                            if (name.asname and name.asname in eval(all_names)) or (not name.asname and name.name in eval(all_names)) \
                            or (name.asname and name.asname not in all_stubs) or (not name.asname and name.name  not in all_stubs) :
                                continue
                            else:
                                alias.append(name)
                        if alias:
                            stmt.names = alias
                            new_imports.append(stmt)
                        else:
                            continue
                elif isinstance(stmt, ast.Import):
                    alias = []
                    for name in stmt.names:
                        if module_name.endswith('.'+name.name) \
                            or module_name == name.name \
                                or 'ctypes' in name.name \
                                    or (name.asname and name.asname in eval(all_names)) \
                                        or (not name.asname and name.name in eval(all_names)) \
                                            or (name.asname and name.asname not in all_stubs) \
                                                or (not name.asname and name.name  not in all_stubs) :
                            continue
                        else:
                            alias.append(name)
                    if len(alias) == 0:
                        continue
                    else:
                        stmt.names = alias
                    new_imports.append(stmt)
                elif isinstance(stmt, ast.Assign):
                    new_target_list = []
                    for target in stmt.targets:
                        if isinstance(target, ast.Name) and target.id not in eval(all_names) and target.id in all_stubs:
                            new_target_list.append(target)
                    if len(new_target_list) > 0:
                        stmt.targets = new_target_list
                        new_imports.append(stmt)
                elif isinstance(stmt, ast.AnnAssign):
                    if isinstance(stmt.target, ast.Name) and stmt.target.id not in eval(all_names) and stmt.target.id in all_stubs:
                        new_imports.append(stmt)
                elif isinstance(stmt, (ast.FunctionDef,ast.AsyncFunctionDef)):
                    if stmt.name not in eval(all_names) and stmt.name in all_stubs:
                        new_imports.append(stmt)
                else:
                    new_imports.append(stmt)
            local_stmts = new_imports
            print(f"import_stmts: {[ast.unparse(stmt) for stmt in local_stmts]}")
            #add_stubs_to_validation_projects
            entity_node = self.node_dict[node_name]
            module_name = '.'.join(Path(entity_node.def_file[:-3]).parts)
            direct_name = node_name.replace(module_name+'.', '')
            name_parts = direct_name.split('.')
            file_path = Validation_Path / (entity_node.def_file)
            #添加类型注释
            source_code = open(file_path).read()
            comment_dict = extract_comment(source_code)
            astNode = ast.parse(source_code)
            has_multiple = len(stub_set) > 1
            offset = []
            for stub in stub_set:
                def_node = ast.parse(stub).body[0]    
                #添加类型注释
                aa = Ann_Adder(name_parts, def_node, has_multiple)
                astNode = aa.visit(astNode)
                offset.extend(aa.added_lines)
            offset.insert(0,(2, len(local_stmts)))
            astNode.body =  astNode.body[:2] + local_stmts + astNode.body[2:]
            astNode = ast.fix_missing_locations(astNode)
            content = ast.unparse(astNode)
            content = restore_comments(content, comment_dict, offset)
            bak_file_path = str(file_path)+  '.bak'
            if not os.path.exists(bak_file_path):
                open(bak_file_path, 'w').write(source_code)
            open(file_path, 'w').write(content)
            changed_dict[file_path] = bak_file_path
            

        msg_list = run_mypy(Validation_Path / self.project)
        logger.info(f"$$$$ ErrMsg $$$$ : {msg_list}\n$$$$ Stub_Dict $$$$ : {stub_dict}")
        if not msg_list:
            for file_path in changed_dict:
                os.remove(changed_dict[file_path])
            return None
        else:
            #分类处理mypy报错
            #恢复文件
            # msg_list = shortest_detail_err(msg_list)
            other_msgs = []
            wrong_functions = set()
            for msg in msg_list:
                root_node = None
                if 'arg-type' in msg['error_code']:
                    logger.debug("Meet Arg-Type Error")
                    try:
                        error_func = re.findall(r'Argument \S+ to "(\w+)"', msg['description'])[0]
                        logger.debug(f"Found Error Function {error_func}")
                    except:
                        other_msgs.append(extract_msg_error_detail(msg))
                        continue
                    try:
                        error_cls = re.findall(r'Argument \S+ to "\w+" of "(\w+)"', msg['description'])[0]
                        error_func = error_cls+'.'+error_func
                        logger.debug(f"Found Error Function {error_func}")
                    except:
                        pass
                    root_node = []                                                                          
                    for node_name in self.node_dict:
                        if node_name.endswith('.'+error_func) or node_name==error_func:
                            #类被调用，会被调用__init__函数
                            if self.node_dict[node_name].category == Entity_Type.class_:
                                target_name = node_name+'.__new__' 
                                if target_name not in self.node_dict.keys():
                                    target_name = node_name+'.__init__'
                                    if target_name not in self.node_dict.keys():
                                        #隐式调用__init__,直接对应类的attribute
                                        try:
                                            err_attr = re.findall(r'Argument "(\w+)" to', msg['description'])[0]
                                            target_name = node_name+'.'+err_attr
                                            if target_name not in self.node_dict:
                                                raise Exception()
                                            else:
                                                if target_name in self.node_dict and self.node_dict[target_name].stub_code is not None and target_name not in stub_dict.keys() and target_name not in previous_wrong_names:
                                                            root_node.append( target_name )
                                        except:
                                            wrong_in_super_cls = False
                                            for v_list in self.class_attr_dict[node_name].values():
                                                    for name in v_list:
                                                        if self.node_dict[name].category == Entity_Type.var and name in stub_dict:
                                                            wrong_in_super_cls = True
                                                            break
                                            if wrong_in_super_cls:
                                                continue
                                            logger.warning(f"找不到冲突方法，清空所有attribute的类型") 
                                            for v_list in self.class_attr_dict[node_name].values():
                                                    for name in v_list:
                                                        if self.node_dict[name].category == Entity_Type.var:
                                                            target_name = name   
                                                        if target_name in self.node_dict and self.node_dict[target_name].stub_code is not None and target_name not in stub_dict.keys() and target_name not in previous_wrong_names:
                                                            root_node.append( target_name )
                                            continue
                            else:   
                                target_name = node_name
                                if target_name in self.node_dict and self.node_dict[target_name].stub_code is not None and target_name not in stub_dict.keys() and target_name not in previous_wrong_names:
                                    root_node.append( target_name )
                            
                if 'override' in msg['error_code']:
                    try:
                        error_func = re.findall(r'Return type "[\S\s]+" of "(\w+)" ',  msg['description'])[0]
                        error_cls = re.findall(r'in supertype "(\w+)"', msg['description'])[0]
                    except:
                        try:
                            error_func = re.findall(r'Signature of "(\w+)" ',  msg['description'])[0]
                            error_cls = re.findall(r'with supertype "(\w+)"', msg['description'])[0]
                        except:
                            try:
                                error_func = re.findall(r'Argument \S+ of "(\w+)" ',  msg['description'])[0]
                                error_cls = re.findall(r'with supertype "(\w+)"', msg['description'])[0]
                            except:
                                other_msgs.append(str(msg))
                                continue
                    logger.debug(f"Super Sub Cls Mismatch: {error_cls} {error_func}")
                    for node_name in stub_dict.keys():
                        if node_name.split('.')[-1] == error_func:
                            sub_cls = '.'.join(node_name.split('.')[:-1])
                            logger.debug(f"Found sub_cls: {sub_cls}")
                            if sub_cls in self.supper_cls_dict:
                                for super_cls in self.supper_cls_dict[sub_cls]:
                                    if super_cls.split('.')[-1] == error_cls:
                                        error_cls = super_cls
                                        break
                            target_name = error_cls+'.'+error_func
                            for node_name in self.node_dict:
                                if (node_name.endswith('.'+target_name) or node_name==target_name) and node_name not in previous_wrong_names and node_name not in stub_dict.keys() and self.node_dict[node_name].stub_code is not None:
                                        root_node = [node_name]
                                        logger.debug(f"Found super_cls: {target_name}")
                                        break
                        if root_node:
                            break
                if root_node:
                    wrong_functions.update(root_node)
                else:
                    if 'in supertype' in msg['description']:
                        other_msgs.append(f"{error_func}的签名与子类的签名不一致，请根据报错修改{error_func}的签名,使其兼容子类签名:{msg['description']}")
                    else:
                        other_msgs.append(extract_msg_error_detail(msg))

                
            
            if other_msgs:
                logger.info(f"Failed cluster: {other_msgs}\n stub_dict: {stub_dict}\n import_stmts: {[ast.unparse(stmt) for stmt in import_stmts]}")
                for file_path in changed_dict:
                    open(file_path, 'w').write(open(changed_dict[file_path]).read())
                    os.remove(changed_dict[file_path])
                if len(other_msgs) > 10:
                    logger.warning(f"Too many other_msgs: {other_msgs}")
                    other_msgs = other_msgs[:10]
                return 2, other_msgs, wrong_functions
            else:
                for file_path in changed_dict:
                    os.remove(changed_dict[file_path])
                return 1, other_msgs, wrong_functions

    
    def check_validation(self, stub_dict:Dict[str, set[str]], import_stmts:List)->Tuple[Optional[Set], List[str]]:
        previous_wrong_names = []
        retry_time = 0
        while True:
            failed_cluster = self.inner_check_validation(stub_dict, import_stmts, previous_wrong_names)
            if not failed_cluster:
                return None, previous_wrong_names
            error_code, other_msgs, wrong_functions = failed_cluster
            if wrong_functions:
                for node_name in wrong_functions:
                    used_name = None
                    if node_name not in self.node_dict:
                        logger.info(f"Node Name {node_name} not found")
                        for correct_name in self.node_dict:
                            if correct_name.endswith('.'+node_name) or correct_name == node_name:
                                logger.info(f"Found Similar Name {correct_name}")
                                used_name = correct_name
                    else:
                        used_name = node_name
                    if not used_name:
                        raise Exception(f"Node Name {node_name} not found")
                    file_path = Validation_Path / (self.node_dict[used_name].def_file)
                    source_code = open(file_path).read()
                    comment_dict = extract_comment(source_code)
                    ast_node = ast.parse(source_code)
                    ast_node,needAny = clear_func_ann(ast_node, self.node_dict[used_name])
                    if needAny:
                        ast_node.body.insert(2, ast.parse("from typing import Any").body[0])
                        offset = [(2,1)]
                    else:
                        offset = []
                    ast_node = ast.fix_missing_locations(ast_node)
                    previous_wrong_names.append(used_name)
                    content = ast.unparse(ast_node)
                    content = restore_comments(content, comment_dict, offset)
                    with open(file_path, 'w') as f:
                        f.write(content)
            elif other_msgs:
                return other_msgs, previous_wrong_names
            retry_time += 1
            if retry_time > 5:
                raise Exception(f"Retry time out when checking {stub_dict}")
    
    def last_try_stub(self, stub_dict:Dict[str, str], import_stmts:List):
        final_stub_dict = {}
        valid_import_stmts = False
        for node_name, stub in stub_dict.items():
            def_node = ast.parse(list(stub)[0]).body[0]
            if isinstance(def_node, ast.Assign):
                errMsg, previous_wrong_names = self.check_validation({node_name:set([ast.unparse(def_node)])},import_stmts)            
                if errMsg or previous_wrong_names:
                    def_node.value = ast.Name(id="Any", ctx=ast.Load()) 
                else:  
                    valid_import_stmts = True
                final_stub_dict[node_name] = set([ast.unparse(def_node)])
            elif isinstance(def_node, ast.AnnAssign):
                errMsg, previous_wrong_names = self.check_validation({node_name:set([ast.unparse(def_node)])},import_stmts)
                if errMsg or previous_wrong_names:
                    def_node.annotation = ast.Name(id="Any", ctx=ast.Load())   
                else:  
                    valid_import_stmts = True
                final_stub_dict[node_name] = set([ast.unparse(def_node)])
            elif isinstance(def_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                default_ann = ast.Name(id="Any", ctx=ast.Load())
                extra_import = ast.parse("from typing import Any").body[0]
                pure_list = [generate_pure_func_nodes(ast.parse(node).body[0], default_ann) for node in stub]
                ann_list = [extract_anno_dict(ast.parse(node).body[0], default_ann) for node in stub]
                for i, pure_node in enumerate(pure_list):
                    anno_dict = ann_list[i]
                    for filed, anno_part in anno_dict.items():
                        if isinstance(anno_part, list):
                            node_part = getattr(pure_node.args, filed)
                            if len(node_part) != len(anno_part):
                                logger.error(f"The number of {filed} in the function  {node_name} definition does not match the number of annotations: {anno_part}, { node_part}")
                            for i, anno in enumerate(anno_part):
                                node_part[i].annotation = anno
                                setattr(pure_node.args, filed, node_part)
                                logger.debug(f"check {node_name}:{ast.unparse(pure_node)}")
                                errMsg, previous_wrong_names = self.check_validation({node_name:set([ast.unparse(node) for node in pure_list])},import_stmts+[extra_import])
                                if errMsg or previous_wrong_names:
                                    node_part[i].annotation = ast.Name(id="Any", ctx=ast.Load()) 
                                else:
                                    valid_import_stmts = True
                                    logger.debug("check pass")
                        elif filed == 'returns':
                            pure_node.returns = anno_part
                            logger.debug(f"check {node_name}:{ast.unparse(pure_node)}")
                            errMsg, previous_wrong_names = self.check_validation({node_name:set([ast.unparse(node) for node in pure_list])},import_stmts+[extra_import])
                            if errMsg or previous_wrong_names:
                                pure_node.returns = ast.Name(id="Any", ctx=ast.Load()) 
                            else:
                                valid_import_stmts = True
                                logger.debug("check pass")
                        else:
                            if anno_part is None:
                                continue
                            node_part = getattr(pure_node.args, filed)
                            if node_part is None:
                                continue
                            node_part.annotation = anno_part
                            setattr(pure_node.args, filed, node_part)
                            logger.debug(f"check {node_name}:{ast.unparse(pure_node)}")
                            errMsg, previous_wrong_names = self.check_validation({node_name:set([ast.unparse(node) for node in pure_list])},import_stmts+[extra_import])
                            if errMsg or previous_wrong_names:
                                node_part.annotation = ast.Name(id="Any", ctx=ast.Load())
                                setattr(pure_node.args, filed, node_part)
                            else:
                                valid_import_stmts = True
                                logger.debug("check pass")
                def_node = pure_node
                final_stub_dict[node_name] = set([ast.unparse(stmt) for stmt in pure_list])
            else:
                raise Exception(f"Unsupported node type: {type(def_node)}:{ast.unparse(def_node)}")
            
        used_imports = [] if not valid_import_stmts else import_stmts
        return final_stub_dict
    
    def extract_dependency(self, cluster:Set[str], LLM_conversation:Optional[List[Dict[str,str]]], known_deps:Set[str]):
        content = LLM_conversation[-1]['content']
        dep_list = []
        lacked_deps = []
        try:
            json_list = re.findall( r'```json\n(.*?)\n```', content, re.DOTALL)[-1]
            JData = json.loads(json_list)
            found = False
            for name in JData:
                if name in self.node_dict:
                    if name not in cluster and name not in known_deps:
                        dep_list.append(name)
                        found = True
                if not found:
                    parent = '.'.join(name.split('.')[:-1])
                    if parent in self.node_dict and parent not in cluster and parent not in known_deps:
                        dep_list.append(parent)
                        found = True
                if not found:
                    root_symbol = name.split('.')[0]
                    for node_name in cluster:
                        def_file = self.node_dict[node_name].def_file
                        module = '.'.join(def_file[:-3].split('/'))
                        full_name = module + '.' + root_symbol
                        if full_name in self.node_dict and full_name not in cluster and full_name not in known_deps:
                            dep_list.append(full_name)
                            found = True
                            break
                if not found:
                    p_parent = '.'.join(name.split('.')[:-2])
                    for node_name in self.node_dict:
                        if node_name in (name, root_symbol, parent, p_parent ) or node_name.endswith('.'+name) or node_name.endswith('.'+parent) or node_name.endswith('.'+root_symbol) or node_name.endswith('.'+p_parent):
                            if node_name not in cluster and node_name not in known_deps and self.node_dict[node_name].category != Entity_Type.class_:
                                dep_list.append(node_name)
                            found = True
                            break
                if not found:
                    logger.debug(f"Failed to find dependency for ***{name}*** in cluster {cluster}")
                    if name.startswith(self.project):
                        lacked_deps.append(name)
            return dep_list,lacked_deps
        except:
            raise Exception(f"Failed to extract stub code from LLM conversation: {content}")
        
    def generate_stub(self, cluster:Set[str], LLM_conversation:Optional[List[Dict[str,str]]]):
        def extract_stub(node: ast.Module, allowed_type_dict, direct_name_dict):
            stub_dict = {}
            environment_stmts = []
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign):
                    def_name = ast.unparse(stmt.target)
                    def_name = def_name.split('.')[-1]
                    if def_name in allowed_type_dict and ast.AnnAssign in allowed_type_dict[def_name]:
                        full_name = direct_name_dict[def_name]
                        stub_dict.setdefault(full_name, set()).add(ast.unparse(stmt))
                    else:
                        environment_stmts.append(stmt)
                elif isinstance(stmt, ast.Assign):            
                    found = False
                    for target in stmt.targets:
                        def_name = ast.unparse(target)
                        def_name = def_name.split('.')[-1]
                        if def_name in allowed_type_dict and ast.Assign in allowed_type_dict[def_name]:
                            full_name = direct_name_dict[def_name]
                            stub_dict.setdefault(full_name, set()).add(ast.unparse(stmt))
                            found = True
                            break
                    if not found:
                        environment_stmts.append(stmt)
                elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    def_name = stmt.name
                    def_name = def_name.split('.')[-1]
                    if def_name in allowed_type_dict and ast.FunctionDef in allowed_type_dict[def_name]:
                        full_name = direct_name_dict[def_name]
                        stmt.body = [ast.Constant(value=Ellipsis)]
                        stub_dict.setdefault(full_name, set()).add(ast.unparse(stmt))
                    else:
                        environment_stmts.append(stmt)
                elif isinstance(stmt, ast.ClassDef):
                    for cls_stmt in stmt.body:
                        if isinstance(cls_stmt, ast.AnnAssign):
                            def_name = ast.unparse(cls_stmt.target)
                            def_name = stmt.name +'.' + def_name.split('.')[-1]
                            if def_name in allowed_type_dict and ast.AnnAssign in allowed_type_dict[def_name]:
                                full_name = direct_name_dict[def_name]
                                stub_dict.setdefault(full_name, set()).add(ast.unparse(cls_stmt))
                            else:
                                environment_stmts.append(cls_stmt)
                        elif isinstance(cls_stmt, ast.Assign):            
                            found = False
                            for target in cls_stmt.targets:
                                def_name = ast.unparse(target)
                                def_name = stmt.name +'.' + def_name.split('.')[-1]
                                if def_name in allowed_type_dict and ast.Assign in allowed_type_dict[def_name]:
                                    full_name = direct_name_dict[def_name]
                                    stub_dict.setdefault(full_name, set()).add(ast.unparse(cls_stmt))
                                    found = True
                                    break
                            if not found:
                                environment_stmts.append(cls_stmt)
                        elif isinstance(cls_stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            def_name = cls_stmt.name
                            def_name = stmt.name +'.' + def_name.split('.')[-1]
                            if def_name in allowed_type_dict and ast.FunctionDef in allowed_type_dict[def_name]:
                                full_name = direct_name_dict[def_name]
                                cls_stmt.body = [ast.Constant(value=Ellipsis)]
                                stub_dict.setdefault(full_name, set()).add(ast.unparse(cls_stmt))
                            # else:
                            #     environment_stmts.append(cls_stmt)
                elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
                # else
                    #normalize import
                    if isinstance(stmt, ast.Import):
                        for alias in stmt.names:
                            if "flake8.src." in alias.name:
                                alias.name = alias.name.replace("flake8.src.", "")
                    if isinstance(stmt, ast.ImportFrom):
                        if stmt.module is not None and 'flake8.src.'in stmt.module:
                            stmt.module = stmt.module.replace("flake8.src.", "")
                    environment_stmts.append(stmt)
            return stub_dict, environment_stmts            

        if not LLM_conversation:
            return {node_name:"" for node_name in cluster},[]
        content = LLM_conversation[-1]['content']
        try:
            stub_code = re.findall( r'```python(.*?)```', content, re.DOTALL)[-1]
        except:
            print(f"Cannot find stub code in {content}---{cluster}")
            exit()
        try:
            astNode = ast.parse(stub_code)
        except Exception as e:
            logger.info(content)
            logger.info(e)
            logger.info(stub_code)
            exit()
        allowed_type_dict = {}
        direct_name_dict = {}
        for node_name in cluster:
            node = self.node_dict[node_name]
            module_name = '.'.join(Path(node.def_file[:-3]).parts)
            direct_name = node_name.replace(module_name+'.', '')
            if '.' in direct_name:
                class_name = direct_name.split('.')[-2]
                direct_name = class_name+'.'+direct_name.split('.')[-1]
            logger.info(f"direct_name: {direct_name}")
            allowed_type = {Entity_Type.var:(ast.AnnAssign, ast.Assign),
                            Entity_Type.func:(ast.FunctionDef, ast.AsyncFunctionDef)}[node.category]
            allowed_type_dict.setdefault(direct_name, allowed_type)
            direct_name_dict.setdefault(direct_name, node_name)
        return extract_stub(astNode,allowed_type_dict, direct_name_dict)
    
    def generate_check_prompt(self, cluster:Set[str]):
        node_list = []
        for node_name in cluster:
            node = self.node_dict[node_name]
            module_name = '.'.join(Path(node.def_file[:-3]).parts)
            snippet_list = []
            for snippet in node.snippet_set:
                if snippet.supper_snippet:
                    dep_str = super_snippet_template.format(function_body = self.snippet_dict[snippet.supper_snippet].snippet)
                else:
                    dep_str =  str({dep:self.node_dict[dep[0]].stub_code for dep in snippet.dependency})
                if node.category == Entity_Type.func:
                    dep_str += possible_types_template.format(possible_types=str(snippet.possible_types))
                snippet_list.append( code_snippet_template.format(code_snippet=snippet.snippet, dep_str=dep_str) )
            
            if node.def_file in self.file_import_dict:
                import_stmts = self.file_import_dict[node.def_file]
            else:
                import_stmts = find_imports(node.def_file)
                self.file_import_dict[node.def_file] = import_stmts
            module_name = '.'.join(node.def_file[:-3].split('/'))
            dep_str =  str({dep:self.node_dict[dep].stub_code for dep in node.appended_dependency})
            node_list.append( node_prompt_template.format(node_name=node.qualified_name, 
                                                          code_snippets=str(snippet_list), 
                                                          import_str=str(import_stmts),
                                                          file_name = module_name,
                                                          all_objs = node.all_objs_in_def_file,
                                                          other_deps = dep_str) )
        prompt = prompt_check_template.format(node_list=node_list,project=self.project).replace(": Unknown_Type","")
        return prompt
    
    def generate_prompt(self, cluster:Set[str]):
        node_list = []
        annotation_list = []
        for node_name in cluster:
            node = self.node_dict[node_name]
            module_name = '.'.join(Path(node.def_file[:-3]).parts)
            direct_name = node_name.replace(module_name+'.', '')
            if '.' in direct_name:
                class_name = direct_name.split('.')[-2]
                if node.category == Entity_Type.func:
                    annotation_template = method_annotation_template.format(class_name = class_name, function_name=node.def_name)
                elif node.category == Entity_Type.var:
                    annotation_template = attr_annotation_template.format(class_name = class_name, var_name=node.def_name)
                else:
                    raise Exception(f"Unknown node category: {node.category}")
            else: 
                if node.category == Entity_Type.func:
                    annotation_template = function_annotation_template.format(function_name=node.def_name)
                elif node.category == Entity_Type.var:
                    annotation_template = var_annotation_template.format(var_name=node.def_name)
                else:
                    raise Exception(f"Unknown node category: {node.category}")
            annotation_list.append(annotation_template)
            snippet_list = []
            for snippet in node.snippet_set:
                if snippet.supper_snippet:
                    dep_str = super_snippet_template.format(function_body = self.snippet_dict[snippet.supper_snippet].snippet)
                else:
                    dep_str =  str({dep:self.node_dict[dep[0]].stub_code for dep in snippet.dependency})
                if node.category == Entity_Type.func:
                    dep_str += possible_types_template.format(possible_types=str(snippet.possible_types))
                snippet_list.append( code_snippet_template.format(code_snippet=snippet.snippet, dep_str=dep_str) )
            
            if node.def_file in self.file_import_dict:
                import_stmts = self.file_import_dict[node.def_file]
            else:
                import_stmts = find_imports(node.def_file)
                self.file_import_dict[node.def_file] = import_stmts
            module_name = '.'.join(node.def_file[:-3].split('/'))
            dep_str =  str({dep:self.node_dict[dep].stub_code for dep in node.appended_dependency})
            node_list.append( node_prompt_template.format(node_name=node.qualified_name, 
                                                          code_snippets=str(snippet_list), 
                                                          import_str=str(import_stmts),
                                                          file_name = module_name,
                                                          all_objs = node.all_objs_in_def_file,
                                                          other_deps = dep_str) )
        prompt = prompt_template.format(node_list=node_list,type_annotation_template='\n'.join(annotation_list)).replace(": Unknown_Type","")
        return prompt
    
    def call_LLM_check(self, batch:List[Set[str]], iteration):
        saved_path = LLM_Result_Path / f"{self.project}" / f"DependencyCheck_{iteration}.json"
        if saved_path.exists():
            with open(saved_path, 'r') as f:
                LLM_Result = json.load(f)
                LLM_Result = {key: value for key, value in LLM_Result.items() if value is not None}
        else:
            LLM_Result = {}
            
        prompt_dict = {}
        for cluster in batch:
            cluster_id = get_cluster_id(cluster)
            prompt_dict[cluster_id] = self.generate_check_prompt(cluster)
            if cluster_id not in LLM_Result:
                raise Exception(f"{cluster_id} not exists in Ieteration {iteration}")
        concurrent_llm_check_dep(prompt_dict, LLM_Result, saved_path)
        return LLM_Result
        
    def call_LLM(self, batch:List[Set[str]], iteration):
        saved_path = LLM_Result_Path / f"{self.project}" / f"TypeInference_{iteration}.json"
        if saved_path.exists():
            with open(saved_path, 'r') as f:
                LLM_Result = json.load(f)
                LLM_Result = {key: value[:2] for key, value in LLM_Result.items() if value is not None}
        else:
            LLM_Result = {}
            
        prompt_dict = {}
        for cluster in batch:
            cluster_id = get_cluster_id(cluster)
            prompt_dict[cluster_id] = self.generate_prompt(cluster)

        concurrent_llm(prompt_dict, LLM_Result, saved_path)
        return LLM_Result
        
if __name__ == "__main__":
    for project in projects:
        # process_project(project)
        try:
            iteration = 0
            target_dir = Validation_Path / project
            if iteration ==  0:
                src_dir = prototype_Path / project
                full_path = EntityGraph_Path / f"{project}.json"
            else:
                src_dir = Validation_Path / f"{project}_{iteration}"
                full_path = EntityGraph_Path / f"{project}_{iteration}.json"
            
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(src_dir, target_dir)

            with open(full_path,'r') as f:
                data = json.load(f)
            graph = Entity_Graph(**data) 
                    
            done = False
            while not done:
                done = True
                
                logger.info("================Iteration:{}================".format(iteration))
                iteration += 1 
                for node in graph.node_dict.values():
                    if node.stub_code is None:
                        # logger.info(f"Node {node.qualified_name} has no stub code")
                        done = False
                graph.preprocess()
                graph.analyze(iteration)
                output_path = EntityGraph_Path / f"{project}_{iteration}.json"
                silent_Write(graph.model_dump_json(indent=4), output_path)   
                target_val_dir = Validation_Path / f"{graph.project}_{iteration}"
                if target_val_dir.exists():
                    shutil.rmtree(target_val_dir)
                shutil.copytree(Validation_Path/ graph.project, target_val_dir)
                
                output_path = EntityGraph_Path / f"{project}_done.json"
                silent_Write(graph.model_dump_json(indent=4), output_path) 
            shutil.rmtree(Validation_Path/ graph.project)
            shutil.copytree(target_val_dir, Validation_Path/ f"{graph.project}")
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, stopping...")
            exit()
        except Exception as e:
            logger.error("Error occurred while processing project", exc_info=True)
            continue