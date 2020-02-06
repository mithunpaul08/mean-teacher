import logging
import os
import git
from mean_teacher.scripts.initializer import Initializer

repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha


LOG=None

class Logger():
    def __init__(self):
        self.LOG=None

    def initialize_logger(self,args):
        log_file_dir=os.path.join(os.getcwd(),args.logs_dir)
        assert os.path.exists(log_file_dir) is True

        log_file_full_path = os.path.join(log_file_dir, 'mean_teacher_' + sha + '.log')

        assert log_file_full_path is not None


        logging.basicConfig(filename=log_file_full_path, filemode='w+')
        self.LOG = logging.getLogger('main')
        LOG=self.LOG
        return LOG