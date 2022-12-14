import ast 
from collections import (
    OrderedDict, defaultdict
)
from tokenize import (
    STRING, NUMBER,
)

OPERATORS = """+, -, **, /, //, %, @,<<,>>,&,|,^,~,:=,<,>,<=,>=,==, !="""
OPERATORS = [op.rstrip().lstrip() for op in OPERATORS.split(",")]

def _find_variables(self):
    """ Generates a list of the variables identified in the code. """
    if self.ast == None:
        self.variables = []
        self.locations["variables"] = {}

        return 

    def gen():
        for node in ast.walk(self.ast):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                yield node.id
            if isinstance(node, ast.arg):
                yield node.arg
                # yield node.name
    
    self.variables = list(OrderedDict.fromkeys(gen()))
    mapping = defaultdict(list)
    
    for idx, t in enumerate(self.tokens):
        if t.src in self.variables:
            mapping[t.src].append(idx)
        
    self.locations["variables"] = mapping

def _find_strings(self):
    ll = [(i, t) for i, t in enumerate(self.tokens) 
          if t.type == STRING and t.src not in self.variables]

    self.strings = [i[1].string for i in ll]
    mapping = defaultdict(list)
    
    for i, t in ll:
        mapping[t.src].append(i)

    self.locations["strings"] = mapping

def _find_numbers(self):
    ll = [(i, t) for i, t in enumerate(self.tokens) if t.type == NUMBER]

    self.numbers = [i[1].string  for i in ll]
    mapping = defaultdict(list)
    
    for i, t in ll:
        mapping[t.src].append(i)

    self.locations["numbers"] = mapping

def _find_operators(self):
    """ Find the operators in the code. """

    ll = [(i, t) for i, t in enumerate(self.tokens)
                      if t.src in OPERATORS]
    
    self.operators = [i[1].src for i in ll]
    mapping = defaultdict(list)

    for i, t in ll:
        mapping[t.src].append(i)
    
    self.locations["operators"] = mapping
    


