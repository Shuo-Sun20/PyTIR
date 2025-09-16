import concurrent.futures
from typing import Dict,List
from pathlib import Path
from type_llm.utils.log_manager import setup_logger
from functools import partial
from type_llm.utils.LLM_Base import _query_llm_with_retry
import re
import ast
import time
from type_llm.utils import silent_JDump
from type_llm.utils.LLM_Helpers import contain_json_list
import traceback

MAX_CONCURRENT = 12
SAVE_LENGTH = 100
RETRY_TIME = 3
logger = setup_logger()

class NameFinder(ast.NodeVisitor):
    def __init__(self):
        self.name_dict = {}
    
    def generic_visit(self, stmt):
        if isinstance(stmt, (ast.FunctionDef,ast.AsyncFunctionDef)):
            stmt.body = [ast.Constant(value=Ellipsis)]
            self.name_dict[stmt.name] = ast.unparse(stmt)
        elif isinstance(stmt, ast.AnnAssign):
            target_name = ast.unparse(stmt.target)
            direct_name = target_name.split('.')[-1]
            self.name_dict[direct_name] = ast.unparse(stmt)
        elif isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                target_name = ast.unparse(target)
                direct_name = target_name.split('.')[-1]
                self.name_dict[direct_name] = ast.unparse(stmt)
        return super().generic_visit(stmt)

def meets_all_needed_names(msg, all_needed_names):
    try:
        stub = re.findall( r'```python(.*?)```', msg, re.DOTALL)[-1]
    except:
        logger.error(f"Python Code not found in {msg}")
        return "Python Code not found in the content"
    try:
        astNode = ast.parse(stub)
    except SyntaxError as e:
        logger.error(f"Invalid Python Code in {msg}")
        return f"AST can not parse the content, ErrorMsg: {str(e)}"
    except Exception as e:
        logger.error(f"Unknown error in {msg}")
        return f"Unknown error occur when AST parsing the Python code in the content"
    nf = NameFinder()
    nf.visit(astNode)
    generated_name = list(nf.name_dict.keys())
    for full_name in all_needed_names:
        found = False
        for name in generated_name:
            if name in full_name: 
                found = True
                break
        if not found:
            logger.error(f"{full_name} not found in {msg}")
            return f"The annotation of {full_name} not found in the content"
    return ""

def concurrent_dep_fix(conservations_dict:Dict[str,List[Dict[str, str]]], history:Dict[str, str] , saved_path: Path):
    thread_map = {}
    idx = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        for prompt_id, conversations in conservations_dict.items():
            logger.debug(f"submitting: {prompt_id}:{conversations}")
            if history.get(prompt_id) is not None \
                and history[prompt_id] != "fail" \
                and (not contain_json_list(history[prompt_id][-1]["content"])):
                idx += 1
                continue 
            thread_map[executor.submit(_query_llm_with_retry, conversations, RETRY_TIME, contain_json_list, prompt_id)] = prompt_id
        logger.info("All Jobs Submitted")
        for future in concurrent.futures.as_completed(thread_map):
            saved_cluster = thread_map[future]
            try:
                result = future.result()
                history[saved_cluster] = result
            except Exception as e:
                logger.error(f"Encounter Error:{e}------{prompt_id}")
                continue
            idx += 1
            print(f"processing {idx} / {len(conservations_dict)}")
            if (idx % SAVE_LENGTH == 0):            
                logger.info(f"Saving checkpoint...")
                time.sleep(3)                    
                silent_JDump(history,saved_path)
                logger.info(f"Saved checkpoint to: {saved_path.resolve()}")
        logger.info(f"Saving final checkpoint...")
        time.sleep(3)
        concurrent.futures.wait(thread_map)
        for future in concurrent.futures.as_completed(thread_map):
            saved_cluster = thread_map[future]
            try:
                result = future.result()
                history[saved_cluster] = result
            except Exception as e:
                logger.error(f"Encounter Error:{e}------{prompt_id}")
        silent_JDump(history,saved_path)
        logger.info(f"Saved final checkpoint to: {saved_path.resolve()}")
        thread_map = {}
    return history

def concurrent_conversation(conservations_dict:Dict[str,List[Dict[str, str]]], history:Dict[str, str] , saved_path: Path):
    thread_map = {}
    idx = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        for prompt_id, conversations in conservations_dict.items():
            all_names = eval(prompt_id)
            is_valid_msg = partial(meets_all_needed_names, all_needed_names = all_names)
            if history.get(prompt_id) is not None \
                and history[prompt_id] != "fail" \
                and (not is_valid_msg(history[prompt_id][-1]["content"])):
                idx += 1
                continue 
            thread_map[executor.submit(_query_llm_with_retry, conversations, RETRY_TIME, is_valid_msg, prompt_id)] = prompt_id
        logger.info("All Jobs Submitted")
        for future in concurrent.futures.as_completed(thread_map):
            saved_cluster = thread_map[future]
            try:
                result = future.result()
                history[saved_cluster] = result
            except Exception as e:
                error_msg = traceback.format_exc()
                print("捕获到异常:\n", error_msg)
                logger.error(f"Encounter Error:{error_msg}------{prompt_id}")
            idx += 1
            print(f"processing {idx} / {len(conservations_dict)}")
            if idx % SAVE_LENGTH == 0:            
                logger.info(f"Saving checkpoint...")
                time.sleep(3)                    
                silent_JDump(history,saved_path)
                logger.info(f"Saved checkpoint to: {saved_path.resolve()}")
        logger.info(f"Saving final checkpoint...")
        time.sleep(3)
        concurrent.futures.wait(thread_map)
        for future in concurrent.futures.as_completed(thread_map):
            saved_cluster = thread_map[future]
            try:
                result = future.result()
                history[saved_cluster] = result
            except Exception as e:
                logger.error(f"Encounter Error:{e}------{prompt_id}")
        silent_JDump(history,saved_path)
        logger.info(f"Saved final checkpoint to: {saved_path.resolve()}")
        thread_map = {}
    return history

def concurrent_llm_universe_dependency(prompt_dict:Dict[str, str], 
                                    history:Dict[str, str] , 
                                    saved_path: Path,
                                    valid_func:callable
                                    ):
    thread_map = {}
    idx = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        for prompt_id, prompt in prompt_dict.items():
            if history.get(prompt_id) is not None \
                and history[prompt_id] != "fail" \
                and (not valid_func(history[prompt_id][-1]["content"]) ):
                idx += 1
                continue
            conservations = [{
            "role": "user",
            "content": prompt,
            }]
            thread_map[executor.submit(_query_llm_with_retry, conservations, RETRY_TIME, valid_func, prompt_id)] = prompt_id
        logger.info("All Jobs Submitted")
        for future in concurrent.futures.as_completed(thread_map):
                saved_cluster = thread_map[future]
                try:
                    result = future.result()
                    history[saved_cluster] = result
                except Exception as e:
                    logger.error(f"Encounter Error:{e}------{saved_cluster}")
                idx += 1
                print(f"processing {idx} / {len(prompt_dict)}")
                if idx % SAVE_LENGTH == 0:          
                    logger.info(f"Saving checkpoint...")
                    time.sleep(3)                    
                    silent_JDump(history,saved_path)
                    logger.info(f"Saved checkpoint to: {saved_path.resolve()}")
        logger.info(f"Saving final checkpoint...")
        time.sleep(3)
        concurrent.futures.wait(thread_map)
        for future in concurrent.futures.as_completed(thread_map):
            saved_cluster = thread_map[future]
            try:
                result = future.result()
                history[saved_cluster] = result
            except Exception as e:
                logger.error(f"Encounter Error:{e}------{prompt_id}")
        silent_JDump(history,saved_path)
        logger.info(f"Saved final checkpoint to: {saved_path.resolve()}")
        thread_map = {}
    return history

def concurrent_llm_check_dep(prompt_dict:Dict[str, str], history:Dict[str, str] , saved_path: Path):
    return concurrent_llm_universe_dependency(prompt_dict, history, saved_path, contain_json_list)
        
def concurrent_llm(prompt_dict:Dict[str, str], history:Dict[str, str] , saved_path: Path):
    thread_map = {}
    idx = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        for prompt_id, prompt in prompt_dict.items():
            all_names = eval(prompt_id)
            is_valid_msg = partial(meets_all_needed_names, all_needed_names = all_names)
            if history.get(prompt_id) is not None and history[prompt_id] != "fail" and (not is_valid_msg(history[prompt_id][-1]["content"]) ):
                idx += 1
                continue
            conservations = [{
            "role": "user",
            "content": prompt,
            }]
            
            thread_map[executor.submit(_query_llm_with_retry, conservations, RETRY_TIME, is_valid_msg, prompt_id)] = prompt_id
        logger.info("All Jobs Submitted")
        for future in concurrent.futures.as_completed(thread_map):
                saved_cluster = thread_map[future]
                try:
                    result = future.result()
                    history[saved_cluster] = result
                except Exception as e:
                    logger.error(f"Encounter Error:{e}------{saved_cluster}")
                idx += 1
                print(f"processing {idx} / {len(prompt_dict)}")
                if idx % SAVE_LENGTH == 0:           
                    logger.info(f"Saving checkpoint...")
                    time.sleep(3)                    
                    silent_JDump(history,saved_path)
                    logger.info(f"Saved checkpoint to: {saved_path.resolve()}")
        logger.info(f"Saving final checkpoint...")
        time.sleep(3)
        concurrent.futures.wait(thread_map)
        for future in concurrent.futures.as_completed(thread_map):
            saved_cluster = thread_map[future]
            try:
                result = future.result()
                history[saved_cluster] = result
            except Exception as e:
                logger.error(f"Encounter Error:{e}------{prompt_id}")
        silent_JDump(history,saved_path)
        logger.info(f"Saved final checkpoint to: {saved_path.resolve()}")
        thread_map = {}
    return history