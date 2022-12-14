""" Wraps useful functionalities to model source code. """

import ast 
from tokenize_rt import src_to_tokens

class SourceCode(object):
    """ Model source code as multiple functions definitions for now. """

    CONSTRUCTS = ["variables", "strings", "operators", "numbers"]

    def __init__(self, source_code) -> None:
        self.source_code = source_code.lstrip() 
        try:
            self.tokens = src_to_tokens(source_code) # The tokens infos
        except (BaseException) as e:
            self.tokens = []
        
        # get the functions, and for each function, get its elements
        
        self._get_ast()

        # map type of construct to a dictionary with location 
        # of every element of that construct
        self.locations = {}
        
        self._find_variables()
        # self._find_strings()
        self._find_operators()
        # self._find_numbers()

    from src.sscm.compilation import _get_ast
    from src.sscm.constructs import (
        _find_variables, _find_strings,
        _find_numbers, _find_operators,
    )

    def is_empty(self):
        return len(self.variables) + len(self.strings) \
             + len(self.operators) + len(self.numbers) == 0




def ast_to_passen_repre(sc_ast):
    """ Transforms a Python AST into the representation
    used for computing the tree edit distance used in 
    the python-edit-distance library 
    """
    adj_list = []
    n_list = []
    i = 0
    
    def get_children(node):
        names = node._fields
        print("names", names)
        return [(f, getattr(node, f)) for f in names]
    
    def dfs(node, i):
        node_name = str(node.__class__.__name__)
        adj_list.append([])
        n_list.append(node_name)
        node_adj_list = []
        for j, c in enumerate(ast.iter_child_nodes(node)):
            dfs(c, i + 1 + j)
            node_adj_list.append(i + 1 + j)
        adj_list[i] = node_adj_list
        
    dfs(sc_ast, i)
    
    return n_list, adj_list