from pyOMT_raw import *

def gen_P(p_s, numX, thresh=1, dataset_name=None):
    '''Given optimized OT class object, this function generates new samples 
       according to the calculated pushed forward distribution. Try dimension 2
       examples for visualization.
    '''
    I_all = torch.empty([2,numX], dtype=torch.long)
    num_bat_x = numX//p_s.bat_size_n
    for ii in range(num_bat_x):
        p_s.pre_cal(ii)
        p_s.cal_measure()
        _, I = torch.sort(p_s.d_U, dim=0,descending=True)
        I_all[0,ii*p_s.bat_size_n:(ii+1)*p_s.bat_size_n].copy_(I[0,:])
        I_all[1,ii*p_s.bat_size_n:(ii+1)*p_s.bat_size_n].copy_(I[1,:])
    
    P = p_s.h_P      
    nm = torch.cat([P, -torch.ones(p_s.num_P,1,dtype=torch.float64)], dim=1)
    nm /= torch.norm(nm,dim=1).view(-1,1)
    cs = torch.sum(nm[I_all[0,:],:] * nm[I_all[1,:],:], 1) #element-wise multiplication
    theta = torch.acos(cs)
    I_gen = I_all[:, theta < thresh]
    _, uni_gen_id = np.unique(I_gen.numpy(), return_index=True, axis=1)
    I_gen = I_gen[:, torch.from_numpy(uni_gen_id)]
    numGen = I_gen.shape[1]
    
    print('OT successfully generated {} samples'.format(numGen))
    rand_w = torch.rand([numGen,1],dtype=torch.float64)
    
    P_gen = torch.mul(P[I_gen[0,:],:], rand_w) + torch.mul(P[I_gen[1,:],:], 1-rand_w)

    if p_s.dim == 2:
        fig2, ax2 = plt.subplots(1, 2, sharey=True, figsize=(14,7))
        fig2.suptitle('Dataset ' + dataset_name, fontsize=16)
        ax2[0].scatter(P[:,0], P[:,1], marker='+', color='orange', label='Real')
        ax2[0].set_title('Groud-truth samples')
        ax2[0].axis('equal')
        ax2[1].scatter(P[:,0], P[:,1], marker='+', color='orange', label='Real')
        ax2[1].scatter(P_gen[:,0], P_gen[:,1], marker='+', color='green', label='Generated')
        ax2[1].set_title('Generated samples')
        ax2[1].axis('equal')
        plt.show()
    
    return P_gen


def gen_P_v2(p_s, numX, thresh=1, dataset_name=None):
    if p_s.num_P != p_s.bat_size_P:
        sys.exit("Error: (num_p) is not equal to (batch_size_P), this method could generate points correctly.")
    if numX % p_s.bat_size_n != 0:
        sys.exit('Error: (numX) is not a multiple of (p_s.bat_size_n)')
    # calculate mu-mass center
    p_s.pre_cal(0)
    p_s.cal_measure(True)
    # generate points
    S_all = torch.empty([numX, p_s.dim], dtype=torch.float, device=torch.device('cuda'))
    I_all = torch.empty([p_s.dim + 1, numX], dtype=torch.long)
    is_generate = torch.ones(numX, dtype=torch.bool)
    num_bat_x = numX // p_s.bat_size_n
    for ii in range(num_bat_x):
        p_s.pre_cal(ii)
        S_all[ii * p_s.bat_size_n:(ii + 1) * p_s.bat_size_n, :].copy_(p_s.d_volP)
        p_s.cal_measure()  
        _, I = torch.sort(p_s.d_U, dim=0, descending=True)  
        for iii in range(p_s.dim + 1):
            I_all[iii, ii * p_s.bat_size_n:(ii + 1) * p_s.bat_size_n].copy_(I[iii, :])
    C = p_s.d_c
    P = p_s.h_P
    nm = torch.cat([P, -torch.ones(p_s.num_P, 1, dtype=torch.float64)], dim=1)
    nm /= torch.norm(nm, dim=1).view(-1, 1)  # shape: (num_p, 1)
    for i_generate in range(p_s.dim):
        cs = torch.sum(nm[I_all[0, :], :] * nm[I_all[i_generate + 1, :], :], 1)
        theta = torch.acos(cs)
        is_generate &= (theta < thresh)
    S_gen = S_all[is_generate, :]
    I_gen = I_all[:, is_generate]
    P_gen = torch.empty(size=S_gen.shape, dtype=torch.float)
    lbd_gen = torch.empty(size=I_gen.shape, dtype=torch.float)

    numGen = I_gen.shape[1]
    print('OT successfully generated {} samples'.format(numGen))

    for i_lbd in range(p_s.dim + 1):
        temp_c = C[I_gen[i_lbd, :], :]
        temp_lbd = 1.0 / torch.norm(S_gen - temp_c, dim=1, keepdim=True)
        lbd_gen[i_lbd:i_lbd + 1, :].copy_(temp_lbd.t())
    lbd_gen /= torch.sum(lbd_gen, dim=0, keepdim=True)
    for i_p in range(p_s.dim + 1):
        P_gen += torch.mul(P[I_gen[i_p, :], :], lbd_gen[i_p:i_p + 1, :].t())

    if p_s.dim == 2:
        fig2, ax2 = plt.subplots(1, 2, sharey=True, figsize=(14, 7))
        fig2.suptitle('Dataset ' + dataset_name, fontsize=16)
        ax2[0].scatter(P[:, 0], P[:, 1], marker='+', color='orange', label='Real')
        ax2[0].set_title('Groud-truth samples')
        ax2[0].axis('equal')
        ax2[1].scatter(P[:, 0], P[:, 1], marker='+', color='orange', label='Real')
        ax2[1].scatter(P_gen[:, 0], P_gen[:, 1], marker='+', color='green', label='Generated')
        ax2[1].set_title('Generated samples')
        ax2[1].axis('equal')
        plt.show()

    return P_gen

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
        gen_P_v2(p_s, bat_size_n, thresh, dataset_name)

        '''clear temporaty files'''
        clear_temp_data()
