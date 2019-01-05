import os, shutil
import pdb

def progbar(curr, total, full_progbar):
    frac = curr/total
    filled_progbar = round(frac*full_progbar)
    print('\r', '#'*filled_progbar + '-'*(full_progbar-filled_progbar), '[{:>7.2%}]'.format(frac), end='')

def clear_folder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

def weights_init(m):
    classname = m.__class__.__name__
    # pdb.set_trace()
    if classname.find('Conv') != -1:
        m.weight.data.uniform_(-0.2, 0.2)
        # pdb.set_trace()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.uniform_(0.5, 1.5)
        m.bias.data.fill_(0)