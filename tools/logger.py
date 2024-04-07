import sys

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = False
        self.buffer = []

    def write(self, message):
        self.terminal.write(message)
        if not self.log:
            self.buffer.append(message)
        else:
            self.log.write(message)
            self.log.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
    
    def redirect(self, filename):
        self.log = open(filename, "w")
        for message in self.buffer:
            self.log.write(message)
        self.log.flush()
        self.buffer = []

    def __del__(self):
        self.log.close()

# create Logger
def create_logger():
    return Logger()

def redirect_logger(filename):
    sys.stdout = create_logger()
    sys.stdout.redirect(filename)