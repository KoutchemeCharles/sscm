import ast 

def _get_ast(self):
    """ Obtain the Abstract Syntax Tree of the code. """
    try:
        self.ast = ast.parse(self.source_code)
        self.compile_error = None
    except (TypeError, SyntaxError, IndentationError) as e:
        self.ast = None
        self.compile_error = str(type(e))[8:-2]
        
