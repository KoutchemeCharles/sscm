import os 
from src.sscm.utils.files import create_dir, save 

def create_save_dir(dataset, save_path):
    """ Create a temporary directory where the Refactory tool is going to
    load the data and perform the repairs. 
    """

    create_dir(save_path, clear=True)
    dataframe = dataset.to_pandas()
    questions = sorted(dataframe["func_name"].unique())
    for i, question in enumerate(questions):
        question_path = os.path.join(save_path, f"question_{i + 1}")
        create_dir(question_path)
        create_ans_folder(dataframe, question, question_path)
        create_code_folder(dataframe, question, i + 1, question_path)
        create_description_file(dataframe, question_path, question)
        # TODO: here somewhere save the description foreach assigment 
        with open(os.path.join(question_path, 'func_name.txt'), 'w') as fp:
            fp.write(str(question))
    
    return questions


def create_description_file(dataframe, question_path, question):
    """ Creates a description.txt file which will contain the assignment/question/code description. """
    
    description = dataframe[dataframe.func_name == question].description.iloc[0]
    save_path = os.path.join(question_path, 'description.txt')
    save(save_path, description)

def create_ans_folder(dataframe, question, question_path):
    """ Takes an func_name, and create the list of inputs and outputs. """
    ans_folder = os.path.join(question_path, "ans")
    create_dir(ans_folder)
    inputs, outputs = get_inputs_outputs(dataframe, question)
    if inputs and outputs:
        get_file_path = lambda f: os.path.join(ans_folder, f)
        for i in range(len(inputs)):
            xxx = "{:03d}".format(i)
            save(get_file_path(f"input_{xxx}.txt"), inputs[i])
            save(get_file_path(f"output_{xxx}.txt"), outputs[i])

def create_code_folder(dataframe, question, question_id, question_path):
    """ Create the folder containing the code part. """
    code_folder = os.path.join(question_path, "code")
    create_dir(code_folder)
    
    # create the correct and wrong subdir
    for name, corr  in zip(["correct", "wrong"], [True, False]):
        corr_folder = os.path.join(code_folder, name)
        create_dir(corr_folder)
        # Get all correct programs in my dataset
        corr_progs = get_all_programs(dataframe, question, corr)
        # Save them 
        get_file_path = lambda f: os.path.join(corr_folder, f)
        for i, code in enumerate(corr_progs):
            xxx = "{:03d}".format(i + 1)
            save(get_file_path(f"{name}_{question_id}_{xxx}.py"), code)
        
        # create reference
        # for the reference we take the first correct program for simplicity 
        if corr:
            ref_folder = os.path.join(code_folder, "reference")
            create_dir(ref_folder)
            # as the reference solution take the solution with the biggest amount of info
              
            df = dataframe[dataframe.func_name == question]

            if "ref_sol" in dataframe.columns:
                ref = df["func_code"].iloc[0] 
            else:
                ref = df["func_code"].value_counts().index[0]

            if len(corr_progs):
                save(os.path.join(ref_folder, "reference.py"), corr_progs[0])
        
    ## Create the global file, which is included just before the assert part
    globals_ = get_globals(dataframe, question)
    if globals_:
        save(os.path.join(code_folder, "global.py"), corr_progs[0])
    
    # additionally, save the name of the original question into a file 
    save(os.path.join(code_folder, "metadata"), question)

def get_globals(df, func_name):
    sub_df = df[(df.func_name == func_name) & (df.test.astype(bool))]
    if sub_df.empty or sub_df.test.iloc[0] == None:
        return ""

    sub_df = df[df.func_name == func_name]
    humaneval = sub_df.test.iloc[0]
    return humaneval[:humaneval.find("assert")].strip()

def get_all_programs(dataframe, func_name, correctness):
    df = dataframe[(dataframe.func_name == func_name) & (dataframe.correct == correctness)]
    return df.func_code.tolist() 

def get_inputs_outputs(df, func_name):
    sub_df = df[(df.func_name == func_name) & (df.test.astype(bool))]
    if sub_df.empty or sub_df.test.iloc[0] == None:
        return None, None

    humaneval = sub_df.test.iloc[0]
    
    test_string = humaneval[humaneval.find("assert"):]
    test_string = test_string.replace("assert", "").strip()

    test_string = test_string.replace(") ==", ")==")
    test_string = test_string.split(" and ")
    #print(test_string)
    inputs, outputs = zip(*[test_case.split(")==") 
                       for test_case in test_string])
    inputs = [i + ')' for i in inputs]


    return inputs, outputs
