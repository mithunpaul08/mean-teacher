import logging
import git
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha

logging.basicConfig(filename='mean_teacher_'+sha+'.log', filemode='w+')
LOG = logging.getLogger('main')