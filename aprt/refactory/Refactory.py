from src.sscm.aprt.refactory.directory import create_save_dir

class RefactoryAPRT():
    """ Allows to run the refactory repair tool and obtain the results. """

    def run(self, save_path, dataset):
        return create_save_dir(dataset, save_path)


# if I do this, then I can move all that part into the dataset for student code part, where
# i can then execute that part in piece and have it once and for all 


