from tokenize_rt import tokens_to_src, src_to_tokens 

def is_parsable(upload):
    """ Checks whether or not a string
    is parsable into python tokens. """
    
    try:
        src_to_tokens(upload)
        return True
    except:
        return False

def find_next_indent(tokens):
    """ Finds the first INDENT token. """
    
    for i, token_info in enumerate(tokens):
        if token_info.name == "INDENT":
            return i
            
    return len(tokens) - 1 if len(tokens) > 0 else 0 

def find_last_dedent(tokens):
    """ Finds the position of the last dedent token scoping
    the function within the tokens. """
    
    n_indent = 1
    for i, token_info in enumerate(tokens):
        
        if token_info.name == "INDENT":
            n_indent += 1
        elif token_info.name == "DEDENT":
            n_indent -= 1
            
        if n_indent == 0:
            return i # simply locate the first double point

    return None

def scope_function(start_index, tokens):
    """ Given a starting index and a list of tokens of a source code,
    find the index at which there is the end of the first function
    encountered. """
    
    start_search = find_next_indent(tokens[start_index:]) + start_index + 1
    end_search = find_last_dedent(tokens[start_search:]) 
    # if there are no dedent, I should just finnish at the last one 
    if end_search is not None:
        end_index = start_search + end_search
    else:
        end_index = start_search + len(tokens)

    return end_index


def find_functions(tokens):
    """ Analyze the sequences of tokens of a source code to
    and yields the list of tokens of the functions defined
    inside that source code. """
    
    for i, token_info in enumerate(tokens):
        if token_info.src == "def":
            start_index = i
            end_index = scope_function(start_index + 1, tokens)
            func_tokens = tokens[start_index: end_index]

            yield func_tokens
            
def get_function_name(tokens):
    """ Given a list of tokenize_rt.tokens of a function, 
    return the name of the function. """
    for t in tokens:
        if t.name == 'NAME' and t.src != "def":
            return t.src 
    return ""

def get_predicted_function(completion, f_name):
    """ Obtain the first defined function in the source code which
    has that function name. """
    
    information = parse_upload(completion)
    for info in information:
        if info["name"] == f_name:
            return info["string"]
    return ""

def parse_upload(upload):
    """ Parse the submitted student code into a list of 
    functions with their tokens. """
    
    try:
        token_infos = src_to_tokens(upload) # The tokens infos
    except (BaseException) as e:
        token_infos = []

    functions_tokens_info = list(find_functions(token_infos))
    
    informations = []
    for tokens_info in functions_tokens_info:
       
        informations.append({
            "tokens": tokens_info,
            "string": tokens_to_src(tokens_info), # this function does not guarantee correct reconstruction
            "name"  : get_function_name(tokens_info)
        })

    return informations


def extract_subroutines(information, main_function_code, function_name):
    """ Extract the subroutines used in the main function code as well
    as the main function code in itself. """
    # super bad way to detect that because the function itself might not contain the right code 
    return [c for c in information if f"{c['name']}(" in main_function_code and c['name'] != function_name] # TODO: replace by regex 
        
            
def extract_relevant_part(solution, function_name):
    """ Extract the relevant part of the student submission for solving
    the specific assignment required. """
    
    # If there are no functions required to be written, extract the whole solution
    if not function_name:
        return solution
    
    # if the function name is specific, look at the code and extract also subroutine
    # if there are not the func_name required, extract nothing. 
    information = parse_upload(solution)

    # first, get the right function, if multiple ones, take the last one written 
    candidates = [info for info in information if info["name"] == function_name]
    
    if len(candidates) == 0:
        return "" # Ignore these students who did not follow the instructions 
    else:
        main_routine = candidates[-1]["string"]
        subroutines = extract_subroutines(information, main_routine, function_name)
        new_string = "\n".join([sr["string"] for sr in subroutines] + [main_routine])
        if not is_parsable(new_string):
            raise ValueError("Routine extraction failed for solution", solution)
        
        return new_string 
        