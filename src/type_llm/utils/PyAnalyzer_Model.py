from pydantic import BaseModel, Field
from type_llm.utils.PyAnalyzer_Enum import *
import json
from typing import Optional, List
from type_llm.utils.log_manager import setup_logger

logger = setup_logger()

class PyAnalyzerVariableSourceLocation(BaseModel):
    start_line: int = Field(alias='startLine', default=-1)
    start_column: int = Field(alias='startColumn', default=-1)
    end_line: Optional[int] = Field(alias='endLine', default=-1)
    end_column: int = Field(alias='endColumn', default=-1)
    
class PyAnalyzerCellSourceLocation(BaseModel):
    start_line: int = Field(alias='startLine', default=-1)
    start_col: int = Field(alias='startCol', default=-1)

class PyAnalyzerCellValues(BaseModel):
    kind: RefKind = RefKind.HasambiguousKind
    in_type_context: bool = False
    
class PyAnalyzerVariableResult(BaseModel):
    id: int
    qualified_name: str = Field(alias='qualifiedName')
    category: EntKind
    location: PyAnalyzerVariableSourceLocation
    file: Optional[str] = Field(alias='File', default=None)
    
class PyAnalyzerCellResult(BaseModel):
    src: int
    dest: int
    values: PyAnalyzerCellValues
    location: PyAnalyzerCellSourceLocation

class PyAnalyzerResult(BaseModel):
    variables: List[PyAnalyzerVariableResult] = Field(default_factory=list)
    cells: List[PyAnalyzerCellResult] = Field(default_factory=list)
                 
def JLoad_PyAnalyzerResult(filename:str)->PyAnalyzerResult:
    with open(filename,'r') as f:
        data = json.load(f)
    PA_Res = PyAnalyzerResult(**data)
    return PA_Res

def JDump_PyAnalyzerResult(data:PyAnalyzerResult ,filename:str):
    JStr = data.model_dump_json(indent=4)
    with open(filename, 'w') as f:
        f.write(JStr)
    