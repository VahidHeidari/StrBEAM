import os
import time



LOG_OUT_DIR = 'LogOutputs'



def Log(msg):
    if not os.path.isdir(LOG_OUT_DIR):                                          # Make output direcotry.
        os.makedirs(LOG_OUT_DIR)
        Log('Logger output dir -> ' + LOG_OUT_DIR)

    log_path = time.strftime('log-%Y%m%d.txt')
    with open(os.path.join(LOG_OUT_DIR, log_path), 'a') as f:
        f.write(str(msg) + '\n')
    print(str(msg))



if __name__ == '__main__':
    print('This is a module!')

