# -*- coding = utf-8 -*-
# @File Name : logger
# @Date : 2022/7/18 14:07
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import sys


# logger class
class Logger(object):
    def __init__(self, filename=None):
        self.filename = filename
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message, terminal=True):
        if terminal:
            self.terminal.write(message)
            self.terminal.write('\n')
        self.log.write(message)
        self.log.write('\n')

    def write_block(self, num_lines):
        for i in range(num_lines):
            line = '*' * 64
            self.write(line, False)

    def flush(self):
        open(self.filename, 'w').close()

    def write_dict(self, dict_obj):
        for key, value in dict_obj.items():
            self.write(key, terminal=False)
            if isinstance(value, dict):
                self.write_dict(value)
            else:
                self.write(str(value), terminal=False)


if __name__ == '__main__':
    logger = Logger()
    logger.flush()
    logger.write('1')
    logger.write('2')
    logger.write_block(5)
