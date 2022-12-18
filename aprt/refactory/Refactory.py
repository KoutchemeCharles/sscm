import os
from src.sscm.aprt.refactory.directory import create_save_dir

class RefactoryAPRT():
    """ Allows to run the refactory repair tool and obtain the results. """

    def run(self, save_path, dataset):
        ds_path = save_path.split("/")[:-1]
        save_path = os.path.join("/", *ds_path, "refactory_format")
        questions = create_save_dir(dataset, save_path)
        return save_path, questions


# if I do this, then I can move all that part into the dataset for student code part, where
# i can then execute that part in piece and have it once and for all 


