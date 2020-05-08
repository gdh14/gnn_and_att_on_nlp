from time import time, sleep, strftime, localtime
import os
        
        
class Logger:
    
    def __init__(self, log_path=None, append_time=True):
        """
        create a logger is log_path is given
        """
        time_str = strftime('%Y-%m-%d-%H-%M-%S',localtime(time()))
        self.log_path = log_path
        if log_path:
            if append_time:
                self.log_file_path = "{}-{}".format(log_path, time_str)                
            else:
                self.log_file_path = self.log_path
            self.write("Start logging - {}".format(time_str))
        
        
    def ensure_path_exist(self, path):
        """
        Make path if it has not been made yet.
        """
        os.makedirs(path, exist_ok=True)
        
    
    def write(self, message):
        if self.log_path:
            with open(self.log_file_path, 'a') as f:
                f.write(message + '\n')
                
    
    def get_log_path(self):
        return self.log_path