import re, ast
import time
import numpy as np
import pandas as pd 
from ast import literal_eval
from collections import OrderedDict, defaultdict
from src.sscm.SourceCode import SourceCode
from src.sscm.extract import parse_upload
from scipy.stats import median_abs_deviation
from datasets import Dataset
from python_minifier import minify
from astor import to_source 

def dataset_apply(dataset, function, column, new_name):

    def f(example):
        example[new_name] = function(column)
    
    return dataset.map(function)

def minify_code(code):
    code = minify(code, rename_locals=False)
    return to_source(ast.parse(code))


def stripComments(code):
    code = str(code)
    return re.sub(r'(?m)^ *#.*\n?', '', code)

def remove_comments(example):
    example["func_code"] = stripComments(example["func_code"])
    return example

def remove_prints(example):
    s = 'print\([\"\'][\w\s]*[\"\']\)'
    example["func_code"] = re.sub(s, '', example["func_code"]) 
    return example 

def has_multiple_functions(code):
    try:
        check = "FunctionDef"
        s = ast.dump(ast.parse(code))
    except:
        check = "def "
        s = code 

    return s.count(check) > 1

def keep_unique_solutions(ds):
    """ Remove duplicate solutions in terms of AST. """

    new_ds = ds.map(add_normalized_ast)
    df = new_ds.to_pandas()
    df = df.drop_duplicates("norm_ast")
    df = df.drop_duplicates("norm_code")
    new_ds = new_ds.select(df.index)

    return new_ds 

def normalize_var_names(code):
    """ Replace variables names in a consistent manner. """
    
    code_ast = ast.parse(code)
    code_string = ast.dump(code_ast)
    
    # Find all the variables in the code
    variables = []
    for node in ast.walk(code_ast):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            variables.append(node.id)
        if isinstance(node, ast.arg):
            variables.append(node.arg)
            
    variables = list(OrderedDict.fromkeys(variables))
    # Maps old variable names to new ones
    new_var_name = {var: f"x_{i}" for i, var in enumerate(variables)}
    
    for var in variables:
        code_string = code_string.replace(f"'{var}'", f"'{new_var_name[var]}'")
        code = code.replace(f"'{var}'", f"'{new_var_name[var]}'")

    return code, code_string

def add_normalized_ast(example, f_name="whole_func_string"):
    norm_ast, norm_code = normalize_var_names(remove_docstring(example[f_name]))
    example["norm_ast"] = norm_ast
    example["norm_code"] = norm_code
    return example

def remove_outliers_with_mad(dataset, column, treshold=2.5):
    df = dataset.to_pandas()
    df = df[[c for c in df.columns if "__index_level_0__" not in c]] 
    df["n_lines"] = df["func_code"].apply(count_lines)
    groups = df.groupby(column, as_index=False, group_keys=False)
    f = lambda gdf: filter_with_mad(gdf, treshold)
    return Dataset.from_pandas(groups.apply(f))

def filter_with_mad(group_df, treshold):
    n_lines = group_df.n_lines
    mad = median_abs_deviation(n_lines)
    mask = np.ones(len(n_lines), dtype=bool)
    if mad:
        mask = ((n_lines - n_lines.median()) / mad) <= treshold
    
    return group_df[mask]

def remove_docstring(code_string):
    """ Remove the docstring part from the full function code. """

    if '"""' in code_string:
        delimiter = '"""'
    elif "'''" in code_string:
        delimiter = "'''"
    else:
        return code_string
    
    doc_start = code_string.find(delimiter)
    doc_end = code_string.find(delimiter, doc_start + 1, -1) + 3

    def remove_part(l, start, end):
        return l[:start] + l[end:].lstrip()

    code_string = remove_part(code_string, doc_start, doc_end)
    
    return code_string

def rest_docstring(code, docstring, test):
    idx = test.find('assert')
    idx = idx + len('assert') if idx != -1 else idx
    
    test = test[idx:].strip()
    
    code = code.splitlines()[0]
    arguments = code[code.find('(') + 1: code.rfind(')')]
    arguments = arguments.split(',')
    
    # For the formation of the docstring, we ignore the case
    # where the student defined optional arguments
    arguments = [a for a in arguments if "=" not in a]
    
    test_cases = test.split(' and ')
    inp, out = test_cases[0].split("==")
    inp, out = inp.strip(), out.strip()
    inp = inp[inp.find('('): inp.rfind(')') + 1]
    inp = literal_eval(inp)
    if len(arguments) == 1:
        inp = [inp]
    
    string = [f":type {arg}: {type(param).__name__}"
               for arg, param in zip(arguments, inp)]
    string.append(f":rtype: {type(literal_eval(out)).__name__}")
    
    doc = "\n".join(string)
    
    formatted_test = ":Example:\n" + "\n".join([("  >>> " + tc) for tc in test_cases])
    full = docstring + '\n\n' + doc + "\n\n" + formatted_test

    return full

def map_no_docstring(example):
    example["docstring"] = ""
    return example

def description_as_docstring(example):
    example["docstring"] = example["description"]
    return example 

def map_augmented_docstring(example): # TODO: different levels of docstring
    example["docstring"] = rest_docstring(example["func_code"],  
                                          example["description"],
                                          example["test"])
    return example 

def include_docstring(docstring, indentation):
    """ Format the docstring such that the identation matches
    the code indentation. 
    """
    lines = docstring.strip().splitlines()
    lines = [l.strip() for l in lines]

    # First line is the summary
    summary = lines[0].strip()
    if len(lines) == 1:
        return f'{indentation}"""{summary}"""'
    
    # Then there is the rest
    lines = lines[:1] + [f"{indentation}{line}" for line in lines[1:]]
    docstring = "\n".join(lines)
    docstring = f'{indentation}"""{docstring}\n{indentation}"""'

    return docstring

def clean_code(code, docstring=""):
    """ Remove comments, initial docstrings, and empty blank lines. 
    Additionally, add a new docstring to the code."""

    code = minify_code(code)
    code  = remove_docstring(code)
    lines = code.strip().split("\n")
    lines = [line.rstrip() for line in lines]
    lines = list(map(stripComments, lines))
    lines = [l for l in lines if l and len(l) > 0]
    idx = next(i for i in reversed(range(len(lines))) if lines[i].startswith("def"))
    lines = lines[idx:]

    # some codes could potentially be single line definitions,
    # we transform these into multiline definitions
    if len(lines) == 1:
        lines = lines[0].split(":")
        lines[0] += ":"

    if docstring:
        n_indents = len(lines[1]) - len(lines[1].lstrip())
        indentation = lines[1][:n_indents]
        lines.insert(1, include_docstring(docstring, indentation))

    return "\n".join(lines)


def map_clean_code(example, column="whole_func_string"):
    example[column] = clean_code(example["func_code"], example["docstring"])
    return example


def find_implementation_start(code):
    lines = code.split("\n")
    marker_indices = ['"""' in l for l in lines]
    end_doc_line = np.nonzero(marker_indices)[0]
    if len(end_doc_line) > 0:
        start = end_doc_line[-1] + 1
    else:
        start = 1
    return start

def remove_additional_generations(completions):
    reg = lambda l: l.startswith("<")
    lines = completions.splitlines()
    extras = list(map(reg, lines))
    extras = list(map(bool, extras))
    idx = len(lines) if True not in extras else extras.index(True)
    return "\n".join(lines[:idx])

def separate_functions(code):

    functions = defaultdict(list)
    source_code = SourceCode(code)
    if source_code.ast:
        code_ast = source_code.ast
        for node in ast.walk(code_ast):
            if isinstance(node, ast.FunctionDef):
                functions[node.name].append(ast.get_source_segment(code, node))
        
    else:
        information = parse_upload(code)
        for info in information:
            functions[info["name"]].append(info["string"])

    return functions

def get_predicted_function(completion, fname):
    functions = separate_functions(completion)
    if not functions or fname not in functions:
        return completion
    return functions[fname][0] # taking the first one 

def get_function_name(code):
    functions = separate_functions(code)
    return list(functions.keys())[0] if functions else ""

def does_compile(code):
    """ Test whether a code does compile. """
    try:
        ast.parse(code)
        return True
    except:
        return False

def count_lines(code):
    return len(code.splitlines())
