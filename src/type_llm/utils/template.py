prompt_template = """
In Python, type annotations are a method for describing the types of variables, function parameters, and return values. 
Below I will provide a series of variables and functions along with their related code snippets. 
Please infer the types of these objects and generate corresponding type annotations.
Please output the type annotations as the following format, retaining the import statements used, and adding necessary import statements.
Important: Place ALL type annotations into a SINGLE unified python code block (do not split into multiple snippets).
```python
{type_annotation_template}
```
Return your explanation for adding type annotations first, then return the code with type annotations added.
Note: Do not use Literal types, please use more general types.
Note: If the object to be inferred is a TypeVar or TypeAlias, or should not have type annotation (for example, an Enum member), ignore the above requirements and just return the original definition statement unchanged.
The list of variables and functions requiring type annotations is as follows:
{node_list}
"""

prompt_template_CN = """
在Python中，pyi文件中的类型注释是用于描述变量、函数参数和返回值的类型的一种方式。下面我将给出一系列变量和函数，及其相关的代码片段列表，请推断这些对象的类型，并生成pyi文件中对应的类型注释。
请按照以下格式生成类型注释，保留用到的import语句，并添加必要的import语句。先返回你对添加类型注释的解释，再返回添加类型注释后的代码。
注意：请不要使用Literal类型，请使用更加general的类型。
注意：如果需要推断类型的对象已经是一个类型了，那么保留原样即可。
```python
{type_annotation_template}
```
需要生成类型注释的变量和函数列表如下：
{node_list}
"""

var_annotation_template = """
<import statements>
{var_name}: <var_type>
"""
function_annotation_template = """
<import statements>
def {function_name}(...)-> <return_type>:
  ...
"""
attr_annotation_template = """
<import statements>
class {class_name}:
  {var_name}: <var_type>
"""
method_annotation_template = """
<import statements>
class {class_name}:
  def {function_name}(...)-> <return_type>:
    ...
"""

node_prompt_template = """
    Name: {node_name},
    Module Location: {file_name},
    Exposed Objects in the same file:{all_objs},
    Import Statements in the same file:{import_str},
    Relevant code snippets：{code_snippets}
    Other potentially relevant dependency information:{other_deps}
"""
node_prompt_base = """
    Name: {node_name},
    Relevant code snippets：{code_snippets}
"""

node_prompt_template_CN = """
    名称：{node_name},
    所在模块:{file_name},
    该文件中暴露的对象包括:{all_objs},
    该文件中中包含的import语句:{import_str},
    相关代码片段列表：{code_snippets}
    其他可能相关的依赖信息:{other_deps}
"""

code_snippet_template = """
code snippet:{code_snippet}
dependencies used in this snippet:{dep_str}
"""

code_snippet_base = """
code snippet:{code_snippet}
"""

code_snippet_template_CN = """
代码片段：{code_snippet}
相关依赖：{dep_str}
"""

dependency_template = """
{var_name}:{stub_code}
"""

possible_types_template = """
Potentially relevant types and their attributes: {possible_types}
"""

possible_types_template_CN = """
定义在项目内部的可能相关的类型及其attributes:{possible_types}
"""

super_snippet_template = """
This snippet is defined in a function, the function body is:{function_body}
"""

super_snippet_template_CN = """
该片段定义在函数中，函数体为:{function_body}
"""

error_template = """
After adding type annotations to the code based on previous responses, mypy checking reported the following errors: {error_msg}. 
Please regenerate type annotations for these objects based on the error messages: {cluster_name}.
Important notes:
1. You may only modify the previously generated code by adjusting type annotations or adding new import statements
2. You must not change any other code
3. First explain your modifications to the type annotations, then provide the updated annotated code
"""

error_template_CN = """
根据回答为代码添加类型注释后，经过mypy检查后报错, 报错信息如下：{error_msg},请根据错误信息重新为下面这些对象生成类型注释:{cluster_name}。
注意：你只能对之前生成的代码进行调整，或者增加新的import语句，不能改变其他代码。
先返回你对添加类型注释的解释，再返回添加类型注释后的代码。
"""

prompt_check_template = """
In Python, type annotations are used to describe the types of variables, function parameters, and return values. 
Below I will provide a series of variables and functions from the {project} project along with their related code snippets.
Please evaluate whether the provided information is sufficient to infer the type annotations.
Important notes:
1. The dependency object names should be specific to class attributes or method names, not just class or module names
2. Return the list of missing dependency object names in the following format. 
```json
[qualified_name_1, qualified_name_2, ...]
```
For objects contained within the {project} project, use their full qualified names prefixed with "{project}.".
First provide your explanation of these dependency requirements, then return the list of dependency object names.
The information provided for generating type annotations is:
{node_list}"""


prompt_check_template_CN = """
在Python中，pyi文件中的类型注释是用于描述变量、函数参数和返回值的类型的一种方式。下面我将给出{project}项目中的一系列变量和函数，及其相关的代码片段列表，请判断提供的信息是否足够用于推断出类型注释的内容。
请将缺少的依赖对象的名称列表以下面的形式返回。其中，包含在{project}项目中的对象请以"{project}."开头的完整qualified name表示。
注意，依赖对象的名称应具体到类的属性或方法的名称，而不是类或模块的名称。
```json
[qualified_name_1, qualified_name_2, ...]
```
先返回你对这些依赖对象需求的解释，再返回依赖对象的名称列表。

用于生成类型注释的信息如下：
{node_list}
"""
reAnalyze_Dep_template ="""
Regarding the previous response, the following dependency objects could not be located: {wrong_name_list}. 
Please verify whether these dependencies exist within the {project} project and regenerate your response."""

reAnalyze_Dep_template_CN = """
在上面的回答中，这些依赖对象无法定位，请确认这些依赖对象是否在{project}项目中，并重新生成回答：{wrong_name_list}。
"""

#unused prompts
clear_wrong_edge = """
我将给你一段Python代码，这段代码位于项目{project_name}的模块{file_name}，来自{qualified_name}。
这段代码通过静态分析预处理，
在每行末尾的注释中标记了该行中引用的本项目中的对象的 **qualified name**，包括变量、函数、类、类属性和类方法等的完整路径。
但由于静态分析精度有限，部分标记可能有误，或来源不正确。

请检查给定代码中每一行的注释标记，从注释中去除 **具有明显问题** 的标记。
对于具有明显问题的标记，请给出具体原因。
对于不确定的标记，不要做任何修改。
若同一行的标记中同时存在不同层级的对象，则全部保留。
注意：所有的标记都是 **qualified name**，即包含模块路径的完整名称。
注意：同一个模块内的对象也需要当前模块的完整路径，`__` 开头的也是合法模块。

先返回你对去除问题标记的解释，再返回去除后的代码和注释标记。

```python
{snippet_code}
```
{qualified_name}所在的上下文中的符号表如下：
{symbol_table}

"""

complete_edge = """
我将给你一段Python代码，这段代码位于项目{project_name}的模块{file_name}，来自{qualified_name}。
这段代码通过静态分析预处理，
在每行末尾的注释中标记了该行中引用的本项目中的对象的 qualified name，包括变量、函数、类、类属性和类方法等的完整路径。
但由于静态分析精度有限，部分标记可能缺失。

请检查每一行中的每一个对象，尝试补全缺失的引用标记。
注意：仅补全能够确定的引用标记，对于不能确定引用的对象，请按下面的方式记录：
若对象是不在符号表之中的符号，以str形式记录其标识符。
若对象是已知引用的未知属性，以str形式记录引用的qualified name和属性名称 as <qualified_name>.<attr>.
注意：所有的标记都是 qualified name，即包含模块路径的完整名称。
注意：同一个模块内的对象也需要当前模块的完整路径，`__` 开头的也是合法模块。

先返回你对新增标记的解释，再返回增加标记后的的代码和注释标记，最后以
```json
[<记录1>，<记录2>，...]
```
的形式返回不确定的对象的记录。

```python
{snippet_code}
```
{qualified_name}所在的上下文中的符号表如下：
{symbol_table}
"""
