import re
from type_llm.utils.log_manager import setup_logger
import ast
from typing import Dict
import json
import re
import tokenize
import io

logger = setup_logger()

pattern = r'[\w.]+'

#validators
def contain_python_code(code: str) -> bool:
    try:
        stub = re.findall( r'```python(.*?)```', code, re.DOTALL)[-1]
    except:
        logger.error(f"Python Code not found in {code}")
        return False
    try:
        astNode = ast.parse(stub)
    except:
        logger.error(f"Invalid Python Code in {code}")
        return False
    return True

def contain_python_code_and_json_list(code: str) -> bool:
    try:
        stub = re.findall( r'```python(.*?)```', code, re.DOTALL)[-1]
        json_list = re.findall( r'```json\n(.*?)\n```', code, re.DOTALL)[-1]
    except:
        logger.error(f"Python Code not found in {code}")
        return False
    try:
        astNode = ast.parse(stub)
        JData = json.loads(json_list)
        for record in JData:
            if not isinstance(record,(list, str)):
                logger.error(f"Invalid JSON in {record}")
                return False
    except:
        logger.error(f"Invalid Python Code in {code}")
        return False
    return True

def contain_json_list(code: str) -> bool:
    try:
        json_list = re.findall( r'```json\n(.*?)\n```', code, re.DOTALL)[-1]
    except Exception as e:
        print(code)
        print(e)
        logger.error(f"Json not found in {code}")
        return "Cannot find Json data in the content"
    try:
        JData = json.loads(json_list)
        for record in JData:
            if not isinstance(record, str):
                logger.error(f"Invalid JSON in {record}")
                return "Json list should contain only string"
    except:
        logger.error(f"Invalid Json Code in {code}")
        return "Cannot parse Json data in the content"
    return ""


def clear_string(s: str) -> str:
    g = tokenize.tokenize(io.BytesIO(s.encode('utf-8')).readline)
    result = []
    for toknum, tokval, _, _, _ in g:
        if toknum == tokenize.STRING:
            #清除字符串，但是保留原本的行号信息
            line_num = tokval.count('\n')
            content = '\n'.join(["pass"] * (line_num+1))
            result.append((toknum, f"\"\"\"{content}\"\"\""))
        else:
            result.append((toknum, tokval))
    return tokenize.untokenize(result).decode('utf-8')

#post analyzers
def collect_type_ann(code_snippet:str):
    collected_dict:Dict[int,set[str]] = {}
    code_snippet = clear_string(code_snippet)
    for i, line in enumerate(code_snippet.split('\n')):
        comment = line.split('#')[-1].strip() if '#' in line else ''
        if comment:
            deps = [i.split(':')[-1].strip() for i in comment.split(',')]
            if deps:
                for dep in deps:
                    true_name = re.findall(pattern,dep)
                    if true_name:
                        collected_dict.setdefault(i, set()).add(true_name[0])
                    # collected_dict.setdefault(i, set()).add(dep)
    return collected_dict

def get_comment_dict(code:str):
    try:
        stub = re.findall( r'```python(.*?)```', code, re.DOTALL)[-1]
        return collect_type_ann(stub)
    except Exception as e:
        return None

def get_comment_dict_and_json_list(code:str):
    try:
        stub = re.findall( r'```python(.*?)```', code, re.DOTALL)[-1]
        json_list = re.findall( r'```json\n(.*?)\n```', code, re.DOTALL)[-1]
        JData = json.loads(json_list)
        JSet = set()
        for i in JData:
            if isinstance(i, str):
                true_name = re.findall(pattern,i)
                if true_name:
                    JSet.add(true_name[0])
                # JSet.add(i)
            else:
                JSet.add(i)
        return collect_type_ann(stub), JSet
    except:
        return None, None

