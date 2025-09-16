from pydantic import BaseModel, Field, field_validator
from typing import Dict, Set, List, Optional
from type_llm.utils.template import clear_wrong_edge, complete_edge
import json
from pathlib import Path
from type_llm.utils.LLM import concurrent_llm_universe_dependency
from type_llm.utils.config import LLM_Result_Path
from type_llm.utils.LLM_Helpers import contain_python_code, contain_python_code_and_json_list, get_comment_dict,get_comment_dict_and_json_list
from type_llm.utils.log_manager import setup_logger

logger = setup_logger()

class Stmt_Location(BaseModel, frozen=True):
    file: str
    line: int

class Snippet_Id(BaseModel, frozen = True):
    defined_name: str
    location: Stmt_Location

# class Dependency_Unit(BaseModel):
#     name: str
#     module_name: str
#     base_class: str
#     snippet: str
#     project: str
#     scope:Dict[str,str]
#     comment_dict:Dict[int,Set[str]]
#     defined_names: Set[str]
#     lacked_symbol_set:List = Field(default_factory=list)
#     @property
#     def annotated_snippet(self):
#         new_lines = []
#         for i, line in enumerate(self.snippet.split('\n')):
#             if i in self.comment_dict:
#                 new_lines.append(f"{line} # {', '.join(self.comment_dict[i])}")
#             else:
#                 new_lines.append(line)
#         annotated_snippet ='\n'.join(new_lines)
#         return annotated_snippet
    
#     def generate_clear_prompt(self):
#         return clear_wrong_edge.format(qualified_name= self.name, 
#                                         file_name=self.module_name, 
#                                         project_name=self.project, 
#                                         snippet_code=self.annotated_snippet,
#                                         symbol_table=self.scope)
    
#     def generate_append_prompt(self):
#         return complete_edge.format(qualified_name= self.name, 
#                                         file_name=self.module_name, 
#                                         project_name=self.project, 
#                                         snippet_code=self.annotated_snippet,
#                                         symbol_table=self.scope)
    
    
# class Dependency_Fixer(BaseModel):
#     project: str
#     node_dict: Dict[Snippet_Id, Dependency_Unit]
    
#     @field_validator('node_dict', mode='before')
#     def check_node_dict(cls, v):
#         new_dic = dict()
#         for snippet_id, snippet in v.items():
#             if isinstance(snippet_id, str):
#                 new_snippet_id = snippet_id.replace("' location=Stmt_Location", "', location=Stmt_Location")
#                 new_dic[eval(f"Snippet_Id({new_snippet_id})", globals(), locals())] = snippet
#             else:
#                 new_dic[snippet_id] = snippet
#         return new_dic
    
#     def clear_wrong_ann(self, saved_file:Path, waiting_list:List[Snippet_Id]):
#         if saved_file.exists():
#             with open(saved_file, 'r') as f:
#                 LLM_Result = json.load(f)
#         else:
#             LLM_Result = {}
#         prompt_comment = {repr(snippet_id): snippet_unit.generate_clear_prompt() for snippet_id, snippet_unit in self.node_dict.items() if snippet_id in waiting_list}
#         concurrent_llm_universe_dependency(prompt_comment, LLM_Result, saved_file, contain_python_code)
#         LLM_Result = {k:v for k,v in LLM_Result.items() if eval(k) in waiting_list and v is not None}
#         return ({eval(snippet_id): get_comment_dict(conversation[0]["content"]) 
#                 for snippet_id, conversation in LLM_Result.items()},
#                 {eval(snippet_id): get_comment_dict(conversation[1]["content"]) 
#                 for snippet_id, conversation in LLM_Result.items()})
        
#     def append_missing_ann(self, saved_file:Path, waiting_list:List[Snippet_Id]):
#         if saved_file.exists():
#             with open(saved_file, 'r') as f:
#                 LLM_Result = json.load(f)
#         else:
#             LLM_Result = {}
#         prompt_comment = {repr(snippet_id): snippet_unit.generate_append_prompt() for snippet_id, snippet_unit in self.node_dict.items() if snippet_id in waiting_list}
#         concurrent_llm_universe_dependency(prompt_comment, LLM_Result, saved_file, contain_python_code_and_json_list)
#         LLM_Result = {k:v for k,v in LLM_Result.items() if eval(k) in waiting_list and v is not None}
#         return {eval(snippet_id): get_comment_dict_and_json_list(conversation[1]["content"]) 
#                 for snippet_id, conversation in LLM_Result.items()}
    
#     def fix_dependency(self, allowed_names:Set[str]):
#         waiting_list = set(self.node_dict.keys())
#         Iter_Id = 0
#         while waiting_list:
#             Iter_Id += 1
#             new_waiting_list = set()
#             clear_file = LLM_Results / self.project / "Dep_Fix" / f"{Iter_Id}_clear.json"
#             append_file = LLM_Results / self.project / "Dep_Fix" / f"{Iter_Id}_append.json"
            
#             #清错边
#             old_dict, clear_dict = self.clear_wrong_ann(clear_file, waiting_list)
#             for snippet_id, comment_dict in clear_dict.items():
#                 if not comment_dict:
#                     continue
#                 if snippet_id not in self.node_dict:
#                     continue
#                 #只清错边，所以 new = new & old
#                 new_comment_dict = {}
#                 for line, comments in comment_dict.items():
#                     comments = comments & old_dict.get(snippet_id, {}).get(line, set())
#                     if comments:
#                         new_comment_dict[line] = comments
#                 self.node_dict[snippet_id].comment_dict = new_comment_dict
            
#             #补漏边
#             append_dict = self.append_missing_ann(append_file, waiting_list)
#             for snippet_id, lack_info in append_dict.items():
#                 if snippet_id not in self.node_dict:
#                     continue
#                 known_symbols = set(self.node_dict[snippet_id].scope.values())  
#                 local_symbols = self.node_dict[snippet_id].defined_names
#                 comment_dict, lacked_symbol_set = lack_info 
#                 if not comment_dict or not lacked_symbol_set:
#                     continue
                
#                 #处理LLM的不稳定性
#                 #1. 如果补的依赖不是符号中已知的实体，则将其放到lacked_symbol_set中
#                 new_comment_dict = {}
#                 for lineno, comment in comment_dict.items():
#                     comment = comment | self.node_dict[snippet_id].comment_dict.get(lineno, set())
#                     existed_symbol = comment & known_symbols
#                     lacked_symbol = comment - known_symbols
#                     lacked_symbol_set.update(lacked_symbol)
#                     if existed_symbol:
#                         new_comment_dict[lineno] = existed_symbol
                    
#                 lacked_symbol_set = lacked_symbol_set - local_symbols
#                 lacked_entity = set()
#                 for lacked_symbol in lacked_symbol_set:
#                     if lacked_symbol  in (self.node_dict[snippet_id].name, self.node_dict[snippet_id].base_class):
#                         logger.debug(f"snippet_id: {snippet_id}, lacked_symbol: {lacked_symbol} is the name of snippet, skip")
#                         continue
#                     elif lacked_symbol in allowed_names:
#                         #2. 如果依赖的符号是已知的合法实体,则加到scope中，并加到waiting_list中
#                         self.node_dict[snippet_id].scope[lacked_symbol] = lacked_symbol
#                         new_waiting_list.add(snippet_id)
#                     else:
#                         based_name = '.'.join(lacked_symbol.split('.')[:-1])
#                         if based_name in allowed_names:
#                             #3. 如果依赖的符号是已知的合法实体的属性,则将(实体，attr)加到缺少列表中
#                             lacked_entity.add((based_name, lacked_symbol.split('.')[-1]))
#                         else:
#                             logger.debug(f"unrecognized symbol: {lacked_symbol}")
                      
#                 self.node_dict[snippet_id].comment_dict = new_comment_dict
#                 if self.node_dict[snippet_id].lacked_symbol_set:
#                     new_lacked_set = self.node_dict[snippet_id].lacked_symbol_set | lacked_entity
#                 else:
#                     new_lacked_set = lacked_entity
#                 if new_comment_dict != old_dict[snippet_id] or new_lacked_set != self.node_dict[snippet_id].lacked_symbol_set:
#                     new_waiting_list.add(snippet_id) 
#                 self.node_dict[snippet_id].lacked_symbol_set = new_lacked_set
#             waiting_list = new_waiting_list
                    
