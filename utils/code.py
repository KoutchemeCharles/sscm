import ast, dis
from io import StringIO
from ast import literal_eval
from collections import defaultdict
from warnings import warn
from src.sscm.SourceCode import SourceCode
from src.sscm.extract import parse_upload
from scipy.stats import median_abs_deviation
from datasets import Dataset
from python_minifier import minify
from astor import to_source 
import textwrap
import numpy as np
import pandas as pd 
from math import ceil 
from tokenize_rt import src_to_tokens, tokens_to_src, Token


def has_multiple_functions(code):
    try:
        check = "FunctionDef"
        s = ast.dump(ast.parse(code))
    except:
        check = "def "
        s = code 

    return s.count(check) > 1

def keep_unique_solutions(ds, code_co, fname_col):
    """ Remove duplicate solutions in terms of a metric. """

    if "DataFrame" not in str(type(ds)):
        df = ds.to_pandas()
    else:
        df = ds

    df["normalized"] = [code_uniqueness(code, func_name) 
                        for code, func_name in df[[code_co, fname_col]].to_numpy()]
    
    def add_representative(sub_df):
        """ Add the representative of the codes having the same appraoch. """
        if not sub_df.empty:
            sub_df["representative"] = sub_df[code_co].value_counts().index[0]
        else:
            sub_df["representative"] = sub_df[code_co]
        return sub_df
    
    
    assert not df.empty
    
    groups = df.groupby([fname_col, "normalized"], as_index=False)
    new_df = groups.apply(add_representative)
    # Now, select only one of the codes which match the representative 
    new_df = new_df[new_df.representative == new_df[code_co]]
    new_df = new_df.drop_duplicates("representative", ignore_index=True, keep='last')
    return Dataset.from_pandas(new_df) 

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


# Docstring 

def remove_docstring(code_string, ret_docstrings=False):
    """ Remove the docstring part from the full function code. """

    # Taken from
    # https://gist.github.com/phpdude/1ae6f19de213d66286c8183e9e3b9ec1
    docstrings = []
    parsed = ast.parse(code_string)
    for node in ast.walk(parsed):
        # let's work only on functions & classes definitions
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            continue

        if not len(node.body):
            continue

        if not isinstance(node.body[0], ast.Expr):
            continue

        if not hasattr(node.body[0], 'value') or not isinstance(node.body[0].value, ast.Str):
            continue
        
        docstrings.append(ast.get_docstring(node))
        node.body = node.body[1:]
    
    code = to_source(parsed)
    if ret_docstrings:
        return docstrings, code
    
    return code 

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

def add_indentation_to_docstring(docstring, indentation):
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

def add_docstring(code, docstring):
    """ Add the given docstring to the code. """
    
    lines = code.split("\n")
    n_indents = len(lines[1]) - len(lines[1].lstrip())
    indentation = lines[1][:n_indents]
    lines.insert(1, add_indentation_to_docstring(docstring, indentation))

    return "\n".join(lines)

def get_code_identation(code):
    code = clean_code(code)
    lines = code.split("\n")
    if len(lines) == 1:
        raise ValueError(code)
        
    n_indents = len(lines[1]) - len(lines[1].lstrip())
    return lines[1][:n_indents] 
    
def clean_code(code, remove_docstring=True):
    """ Minify and clean a source code.
    
    Remove comments, initial docstrings, and empty blank lines. 
    Additionally, add a new docstring to the code.
    """

    code = minify(code, rename_locals=False, remove_literal_statements=remove_docstring)
    return to_source(ast.parse(code)).strip()
    
# Special for handling LLM model generations

def find_implementation_start(code):
    lines = code.split("\n")
    marker_indices = ['"""' in l for l in lines]
    end_doc_line = np.nonzero(marker_indices)[0]
    if len(end_doc_line) > 0:
        start = end_doc_line[-1] + 1
    else:
        # it's not necessarily line 1, could be other codes afterwards... 
        pos = [i for i, l in enumerate(lines) if l.startswith("def")][-1]
        start = pos + 1

    return start

def remove_additional_generations(completions):
    reg = lambda l: bool(l.startswith("<") or l.strip().startswith("#"))
    lines = completions.splitlines()
    extras = list(map(reg, lines))
    extras = list(map(bool, extras))
    lines = [l for b, l in zip(extras, lines) if not b]
    # idx = len(lines) if True not in extras else extras.index(True)
    # also remove the first line with the file ext
    return "\n".join(lines)

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
        exec(code)
        return True
    except:
        return False

# Statistics 
def count_lines(code):
    return len(code.splitlines())

# Dataset application
def dataset_apply(func, columns, new_name=""):

    if type(columns) == str:
        columns = [columns]
    new_name = new_name if new_name else columns[0]

    def f(example):
        args = [example[c] for c in columns]
        example[new_name] = func(*args)
        return example
    
    return f


# Execution

def get_code_executables(code):
    """ Get the defined in the functions
    
    Parameters
    ----------
    code: string
        Contains the definition of the function (as well as possible auxiliary functions definitions)
        we want to execute. 
    
    Returns
    -------
    d: Dict
        Functions and 
        
    """
    dictionary = {}
    string = textwrap.dedent(code)
    exec(string, dictionary, dictionary)
    
    return dictionary

def disassemble(func):
    """ Diassemble a function into a set of instructions. """

    output = StringIO()
    dis.dis(func, file=output)
    lines = output.getvalue().splitlines()
    lines = [l[3:].strip() for l in lines]
    return "\n".join(lines)

def code_uniqueness(code, fname, method="dumped_ast"):
    """ Returns a normalized version of the code which could be
    used later to compare functions equivalence. """
    
    # Inspirted by
    # https://stackoverflow.com/questions/20059011/check-if-two-python-functions-are-equal
    
    if get_function_name(code) != fname:
        warn(f"Code {code} canot have a unique value")
        return None

    executables = get_code_executables(code)
    if fname not in executables:
        raise ValueError(f"Function {code} could not be obtained")
    func = executables[fname]
    if func is None:
        raise ValueError(f"Function {code} could not be obtained")
        
    variables = func.__code__.co_varnames
    new_var_name = {var: f"x_{i}" for i, var in enumerate(variables)}

    if method == "bytecode":
        func.__code__ = func.__code__.replace(co_varnames=tuple(new_var_name.values()))
        return func.__code__.co_code 
    elif method == "dumped_ast":
        dumped = ast.dump(ast.parse(code))
        for var in variables:
            dumped = dumped.replace(f"'{var}'", f"'{new_var_name[var]}'")
        return dumped
    elif method == "dis":
        func.__code__ = func.__code__.replace(co_varnames=tuple(new_var_name.values()))
        return disassemble(func)
    else:
        raise ValueError("uknown method")


def keep_percentage(dataset, percentage, code_col, group_col, ref_sol_col=None):
    df = dataset.to_pandas()
    if type(percentage) == int and percentage > 0:
        f  = lambda sub_df: sub_df.head(ceil(len(sub_df)*(percentage/100)))
    elif (percentage == 0) or (percentage == "ref_sol"):
        if ref_sol_col: 
            f = lambda sub_df: sub_df.loc[sub_df[ref_sol_col], code_col].iloc[0]
        else: # select the most common solution as the "reference solution" 
            def f(sub_df):
                repre = sub_df[code_col].value_counts().index[0]
                return sub_df[sub_df[code_col] == repre].iloc[0]
    else:
        f = lambda sub_df: sub_df
    
    df = df.groupby(group_col, as_index=False).apply(f).reset_index(drop=True)

    return Dataset.from_pandas(df, preserve_index=False)





def match_variables(source, destination, func_name):
    """ Matches variables from destination with variables from source. """
    
    # Take all variables from source (incorrect code)
    # find their counterparts in destination
    src_func = get_code_executables(source)[func_name]
    src_arguments = src_func.__code__.co_varnames[:src_func.__code__.co_argcount]
     
    # match each argument in source with each argument in destination
    dest_func = get_code_executables(destination)[func_name]
    dest_arguments = dest_func.__code__.co_varnames[:dest_func.__code__.co_argcount]
    
    new_var_names = src_arguments[:len(dest_arguments)]
    new_var_names = {d: s for s, d in zip(src_arguments, dest_arguments)}
    
    dest_tokens = src_to_tokens(destination)
    
    offset = 0
    new_dest_tokens = []
    for t in dest_tokens:
        src = new_var_names.get(t.src, t.src) if t.name == "NAME" else t.src 
        new_token = Token(name=t.name, src=src, utf8_byte_offset=offset)
        new_dest_tokens.append(new_token)
        offset += len(src)
    
    return tokens_to_src(new_dest_tokens)
    
