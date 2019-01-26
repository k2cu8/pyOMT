import sys

import torch
from torch.utils.data import DataLoader
from pyculib import rand
from numba import cuda
import pdb


import torchvision
from torchvision import transforms
import pyOMT_utils as ut
import numpy as np
import os

from torch import nn
import torch.optim as optim

import matplotlib.pyplot as plt



#!add cuda array interface to torch.Tensor object
def torch_cuda_array_interface(tensor):
    """Array view description for cuda tensors.

    See:
    https://numba.pydata.org/numba-doc/latest/cuda/cuda_array_interface.html
    """

    if not tensor.device.type == "cuda":
        # raise AttributeError for non-cuda tensors, so that
        # hasattr(cpu_tensor, "__cuda_array_interface__") is False.
        raise AttributeError("Tensor is not on cuda device: %r" % tensor.device)

    if tensor.requires_grad:
        # RuntimeError, matching existing tensor.__array__() behavior.
        raise RuntimeError(
            "Can't get __cuda_array_interface__ on Variable that requires grad. "
            "Use var.detach().__cuda_array_interface__ instead."
        )

    typestr = {
        torch.float16: "<f2",
        torch.float32: "<f4",
        torch.float64: "<f8",
        torch.uint8: "|u1",
        torch.int8: "|i1",
        torch.int16: "<i2",
        torch.int32: "<i4",
        torch.int64: "<i8",
    }[tensor.dtype]

    itemsize = tensor.storage().element_size()

    shape = tensor.shape
    strides = tuple(s * itemsize for s in tensor.stride())
    data = (tensor.data_ptr(), False)

    return dict(typestr=typestr, shape=shape, strides=strides, data=data, version=0)

torch.Tensor.__cuda_array_interface__ = property(torch_cuda_array_interface)

#redefine pyculib generator
#!input_P: input file list of P
class pyOMT_raw():	
    def __init__ (self, h_P, numP, dim_y, maxIter, lr, bat_size_P, bat_size_n):
        self.h_P = h_P
        self.numP = numP
        self.dim_y = dim_y
        self.maxIter = maxIter
        self.lr = lr
        self.bat_size_P = bat_size_P
        self.bat_size_n = bat_size_n

        if numP % bat_size_P != 0:
        	sys.exit('Error: (numP) is not a multiple of (bat_size_P)')

        self.num_bat_P = numP // bat_size_P
        #!internal variables
        self.d_G_z = torch.empty(self.bat_size_n*self.dim_y, dtype=torch.float, device=torch.device('cuda'))
        self.d_volP = torch.empty((self.bat_size_n, self.dim_y), dtype=torch.float, device=torch.device('cuda'))
        self.d_h = torch.zeros(self.numP, dtype=torch.float, device=torch.device('cuda'))
        self.d_delta_h = torch.zeros(self.numP, dtype=torch.float, device=torch.device('cuda'))
        self.d_ind = torch.empty(self.bat_size_n, dtype=torch.long, device=torch.device('cuda'))
        self.d_ind_val = torch.empty(self.bat_size_n, dtype=torch.float, device=torch.device('cuda'))
        
        self.d_ind_val_argmax = torch.empty(self.bat_size_n, dtype=torch.long, device=torch.device('cuda'))
        self.d_tot_ind = torch.empty(self.bat_size_n, dtype=torch.long, device=torch.device('cuda'))
        self.d_tot_ind_val = torch.empty(self.bat_size_n, dtype=torch.float, device=torch.device('cuda'))
        self.d_g = torch.zeros(self.numP, dtype=torch.float, device=torch.device('cuda'))
        self.d_g_sum = torch.zeros(self.numP, dtype=torch.float, device=torch.device('cuda'))
        self.d_adam_m = torch.zeros(self.numP, dtype=torch.float, device=torch.device('cuda'))
        self.d_adam_v = torch.zeros(self.numP, dtype=torch.float, device=torch.device('cuda'))

        #!temp variables
        self.d_U = torch.empty((self.bat_size_P, self.bat_size_n), dtype=torch.float, device=torch.device('cuda'))
        self.d_temp_h = torch.empty(self.bat_size_P, dtype=torch.float, device=torch.device('cuda'))
        self.d_temp_P = torch.empty((self.bat_size_P, self.dim_y), dtype=torch.float, device=torch.device('cuda'))

        print('Allocated GPU memory: {}MB'.format(torch.cuda.memory_allocated()/1e6))
        print('Cached memory: {}MB'.format(torch.cuda.memory_cached()/1e6))


        #!debug variable
        # self.d_P = torch.rand((self.numP, self.dim_y), dtype=torch.float, device=torch.device('cuda'))



    def pre_cal(self,count):
        '''prepare random feed w. qrnd'''
        d_y = cuda.as_cuda_array(self.d_G_z)
        qrng = rand.QRNG(rndtype=rand.QRNG.SCRAMBLED_SOBOL32, ndim=self.dim_y, offset=count*self.bat_size_n)
        qrng.generate(d_y)
        self.d_volP.copy_(self.d_G_z.view(self.dim_y, self.bat_size_n).t())
        self.d_volP.add_(-0.5)
        # self.d_volP.uniform_(-0.5,0.5)


    def cal_measure(self):
        self.d_tot_ind_val.fill_(-1e30)
        self.d_tot_ind.fill_(-1)
        i = 0     
        while i < self.numP // self.bat_size_P:
            temp_P = self.h_P[i*self.bat_size_P:(i+1)*self.bat_size_P]
            temp_P = temp_P.view(temp_P.shape[0], -1)	
                
            '''U=PX+H'''
            self.d_temp_h = self.d_h[i*self.bat_size_P:(i+1)*self.bat_size_P]
            self.d_temp_P.copy_(temp_P)
            #self.d_temp_P = torch.narrow(self.d_P,0,i*self.bat_size_P,self.bat_size_P)
            torch.mm(self.d_temp_P, self.d_volP.t(),out=self.d_U)
            torch.add(self.d_U, self.d_temp_h.expand([self.bat_size_n, -1]).t(), out=self.d_U)
            '''compute max'''
            torch.max(self.d_U, 0, out=(self.d_ind_val, self.d_ind))
            '''add P id offset'''
            self.d_ind.add_(i*self.bat_size_P)
            '''store best value'''
            torch.max(torch.stack((self.d_tot_ind_val, self.d_ind_val)), 0, out=(self.d_tot_ind_val, self.d_ind_val_argmax))
            self.d_tot_ind = torch.stack((self.d_tot_ind, self.d_ind))[self.d_ind_val_argmax, torch.arange(self.bat_size_n)] 
            '''add step'''
            i = i+1
            
        '''calculate histogram'''
        self.d_g.copy_(torch.bincount(self.d_tot_ind, minlength=self.numP))
        self.d_g.div_(self.bat_size_n)
        

    def update_h(self):
        self.d_g -= 1./self.numP
        self.d_adam_m *= 0.9
        self.d_adam_m += 0.1*self.d_g
        self.d_adam_v *= 0.999
        self.d_adam_v += 0.001*torch.mul(self.d_g,self.d_g)
        torch.mul(torch.div(self.d_adam_m, torch.add(torch.sqrt(self.d_adam_v),1e-8)),-self.lr,out=self.d_delta_h)
        torch.add(self.d_h, self.d_delta_h, out=self.d_h)
        '''normalize h'''
        self.d_h -= torch.mean(self.d_h)


    def run_gd(self, last_step=0, num_bat=1):
        g_ratio = 1e20
        best_g_norm = 1e20
        curr_best_g_norm = 1e20
        steps = 0
        count_bad = 0
        dyn_num_bat_n = num_bat
        h_file_list = []
        m_file_list = []
        v_file_list = []

        while(steps <= self.maxIter):
            self.d_g_sum.fill_(0.)
            for count in range(dyn_num_bat_n):
                self.pre_cal(count)
                self.cal_measure()
                torch.add(self.d_g_sum, self.d_g, out=self.d_g_sum)
                # ut.progbar(count+1,dyn_num_bat_n, 20)
            # print(' ')
            torch.div(self.d_g_sum, dyn_num_bat_n, out=self.d_g)			
            self.update_h()

            g_norm = torch.sqrt(torch.sum(torch.mul(self.d_g,self.d_g)))
            num_zero = torch.sum(self.d_g == -1./self.numP)

            torch.abs(self.d_g, out=self.d_g)
            g_ratio = torch.max(self.d_g)*self.numP
            
            print('[{0}/{1}] Max absolute error ratio: {2:.3f}. g norm: {3:.6f}. num zero: {4:d}'.format(
                steps, self.maxIter, g_ratio, g_norm, num_zero))

            if g_ratio < 1e-2:
                torch.save(self.d_h, './h_final.pt')
                return


            torch.save(self.d_h, './h/{}.pt'.format(steps+last_step))
            torch.save(self.d_adam_m, './adam_m/{}.pt'.format(steps+last_step))
            torch.save(self.d_adam_v, './adam_v/{}.pt'.format(steps+last_step))
            h_file_list.append('./h/{}.pt'.format(steps+last_step))
            m_file_list.append('./adam_m/{}.pt'.format(steps+last_step))
            v_file_list.append('./adam_v/{}.pt'.format(steps+last_step))
            if len(h_file_list)>5:
                if os.path.exists(h_file_list[0]):
                    os.remove(h_file_list[0])
                h_file_list.pop(0)
                if os.path.exists(v_file_list[0]):
                    os.remove(v_file_list[0])
                v_file_list.pop(0)
                if os.path.exists(m_file_list[0]):
                    os.remove(m_file_list[0])
                m_file_list.pop(0)

            if g_norm <= curr_best_g_norm:
                curr_best_g_norm = g_norm
                count_bad = 0
            else:
                count_bad += 1
            if count_bad > 20:
                dyn_num_bat_n *= 2
                print('bat_size_n has increased to {}'.format(dyn_num_bat_n*self.bat_size_n))
                count_bad = 0
                curr_best_g_norm = 1e20

            steps += 1


    def set_h(self, h_tensor):
        self.d_h.copy_(h_tensor)

    def set_adam_m(self, adam_m_tensor):
        self.d_adam_m.copy_(adam_m_tensor)

    def set_adam_v(self, adam_v_tensor):
        self.d_adam_v.copy_(adam_v_tensor)

    # def T_map(self, x):
    #     numX = x.shape[0]
    #     x = x.view(numX,-1)
    #     result_id = torch.empty([numX], dtype=torch.long, device=torch.device('cuda'))
    #     result = torch.empty([numX, dim_y], dtype=torch.float, device=torch.device('cuda'))
    #     for ii in range(numX//500 + 1):			
    #         x_bat = x[ii*500 : min((ii+1)*500, numX)]
    #         tot_ind_val = torch.empty([x_bat.shape[0]],dtype=torch.float, device=torch.device('cuda'))
    #         tot_ind_val.fill_(-1e30)
    #         tot_ind = torch.empty([x_bat.shape[0]],dtype=torch.long, device=torch.device('cuda'))
    #         tot_ind.fill_(-1)
    #         ind_val_argmax = torch.empty([x_bat.shape[0]],dtype=torch.long, device=torch.device('cuda'))
    #         ind_val_argmax.fill_(-1)

    #         data_iter = iter(self.dataloader)
    #         i = 0
    #         while i < len(self.dataloader):
    #             temp_P,_ = data_iter.next()
    #             temp_P = temp_P.view(temp_P.shape[0],-1)	
                
    #             '''U=PX+H'''
    #             self.d_temp_h = self.d_h[i*self.bat_size_P:(i+1)*self.bat_size_P]
    #             self.d_temp_P.copy_(temp_P)
    #             U = torch.mm(self.d_temp_P,x_bat.t())
    #             U = torch.add(U,self.d_temp_h.expand([x_bat.shape[0],-1]).t())
    #             '''compute max'''
    #             ind_val, ind = torch.max(U,0)
    #             curr_result = self.d_temp_P[ind]

    #             ind.add_(i*self.bat_size_P)
                
    #             torch.max(torch.stack((tot_ind_val,ind_val)),0,out=(tot_ind_val,ind_val_argmax))
    #             tot_ind = torch.stack((tot_ind,ind))[ind_val_argmax, torch.arange(x_bat.shape[0])] 

                
    #             result[ii*500 : min((ii+1)*500, numX)] = torch.cat(
    #                 (result[ii*500 : min((ii+1)*500, numX)],curr_result), dim=0)[
    #                 ind_val_argmax * x_bat.shape[0] + torch.arange(x_bat.shape[0]).cuda()]
    #             i+=1
    #         result_id[ii*500 : min((ii+1)*500, numX)] = tot_ind
    #     return result, result_id


def load_last_file(path, file_ext):
    if not os.path.exists(path):
        os.makedirs(path)
        return None, None
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    file_ids = [(int(f.split('.')[0]), os.path.join(path,f)) for f in files]
    if not file_ids:
        return None, None
    else:
        last_f_id, last_f = max(file_ids, key=lambda item:item[0])
        print('Last' + path + ': ', last_f_id)
        return last_f_id, last_f

def train_omt(p_s, num_bat=1):
    last_step = 0
    '''load last trained model parameters and last omt parameters'''
    h_id, h_file = load_last_file('./h', '.pt')
    adam_m_id, m_file = load_last_file('./adam_m', '.pt')
    adam_v_id, v_file = load_last_file('./adam_v', '.pt')
    if h_id != None:
        if h_id != adam_m_id or h_id!= adam_v_id:
            sys.exit('Error: h, adam_m, adam_v file log does not match')
        elif h_id != None and adam_m_id != None and adam_v_id != None:
            last_step = h_id
            p_s.set_h(torch.load(h_file))
            p_s.set_adam_m(torch.load(m_file))
            p_s.set_adam_v(torch.load(v_file))

    '''run gradient descent'''
    p_s.run_gd(last_step=last_step, num_bat=num_bat)

    '''record result'''
    np.savetxt('./h_final.csv',p_s.d_h.cpu().numpy(), delimiter=',')

def gen_P(p_s, numX, thresh):
    I_all = torch.empty([2,numX], dtype=torch.long)
    num_bat_x = numX//p_s.bat_size_n
    for ii in range(num_bat_x):
        p_s.pre_cal(ii)
        p_s.cal_measure()
        _, I = torch.sort(p_s.d_U, dim=0,descending=True)
        I_all[0,ii*p_s.bat_size_n:(ii+1)*p_s.bat_size_n].copy_(I[0,:])
        I_all[1,ii*p_s.bat_size_n:(ii+1)*p_s.bat_size_n].copy_(I[1,:])
    '''plot I'''
    fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
    axs[0].hist(I_all[0,:].numpy(), bins=p_s.numP)
    axs[1].hist(I_all[1,:].numpy(), bins=p_s.numP)
    
    
    P = p_s.h_P  
    
    nm = torch.cat([P, -torch.ones(p_s.numP,1,dtype=torch.float64)], dim=1)
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

    fig2, ax2 = plt.subplots()
    ax2.scatter(P[:,0], P[:,1], marker='+', color='orange', label='Real')
    ax2.scatter(P_gen[:,0], P_gen[:,1], marker='+', color='green', label='Generated')
    plt.show()
    

if __name__ == '__main__':
    '''args for omt'''
    input_P = 'data/25gaussians.csv'
    h_P = torch.from_numpy(np.genfromtxt(input_P,delimiter=','))
    numP = 256
    dim_y = 2
    maxIter = 0
    lr = 1e-4
    bat_size_P = 256
    bat_size_n = 200000

    p_s = pyOMT_raw(h_P, numP, dim_y, maxIter, lr, bat_size_P, bat_size_n)
    '''train omt'''
    train_omt(p_s, 1)

    '''generate new samples'''
    #use threshold 0.09 for Swissroll
    #use threshold 0.4 for 8 Gaussians
    gen_P(p_s, bat_size_n, 0.2)
    
    


            