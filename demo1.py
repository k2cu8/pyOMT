from pyOMT_raw import *

if __name__ == '__main__':
    dataset_names = ['25gaussians','8gaussians','swissroll']
    
    num_P = 256
    dim = 2
    max_iter = 20000
    lr = 1e-4
    bat_size_P = 256
    bat_size_n = 10000
    
    for dataset_name in dataset_names:
        if dataset_name == '25gaussians':
            lr = 1e-4
            thresh = 0.2
        elif dataset_name == '8gaussians':
            lr = 2e-5
            thresh = 0.4
        elif dataset_name == 'swissroll':
            thresh = 0.09
        
        input_P = 'data/' + dataset_name + '.csv'
        h_P = torch.from_numpy(np.genfromtxt(input_P,delimiter=','))
        p_s = pyOMT_raw(h_P, num_P, dim, max_iter, lr, bat_size_P, bat_size_n)

        '''train omt'''
        train_omt(p_s, 1)

        '''generate new samples'''
        gen_P(p_s, bat_size_n, thresh, dataset_name)

        '''clear temporaty files'''
        clear_temp_data()