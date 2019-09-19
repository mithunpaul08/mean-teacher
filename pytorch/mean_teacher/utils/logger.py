import logging

class Logger():
    def __init__(self, args):

        logging.basicConfig(filename='mean_teacher.log', filemode='w+')
        self.LOG = logging.getLogger('main')
        self.LOG.setLevel(args.log_level)

    def get_logger(cls):
        return cls.LOG