import sys

import torch
from torch.utils.data import DataLoader
from pyculib import rand
from numba import cuda
import P_loader
import pdb
from models_64x64 import Generator
from torchvision import transforms
import pyOMT_utils as ut
import numpy as np
import os

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
class pyOMT_simple():	
	def __init__ (self, input_P, d_G_model, numP, dim_y, dim_z, maxIter, lr, bat_size_P, bat_size_n):
		self.dataset = P_loader.P_loader(root=input_P,transform=transforms.ToTensor())
		self.dataloader = DataLoader(self.dataset, batch_size=bat_size_P, shuffle=False, pin_memory=True, drop_last=True, num_workers = 8)
		self.G_set = P_loader.P_loader(root='./data/G_z',loader=P_loader.G_z_loader)
		self.G_loader = DataLoader(self.G_set, batch_size=bat_size_n//500, shuffle=False, drop_last=True, num_workers = 8)
		self.d_G_model = d_G_model
		self.numP = numP
		self.dim_z = dim_z
		self.dim_y = dim_y
		self.maxIter= maxIter
		self.lr = lr
		self.bat_size_P = bat_size_P
		self.bat_size_n = bat_size_n

		if numP % bat_size_P != 0:
			sys.exit('Error: (numP) is not a multiple of (bat_size_P)')

		if bat_size_n % 500 != 0:
			sys.exit('Error: (bat_size_n) must be a multiple of 500')
		self.num_bat_P = numP // bat_size_P
		#!internal variables
		self.d_z = torch.empty(self.bat_size_n*self.dim_z, dtype=torch.float, device=torch.device('cuda'))
		self.d_G_z = torch.empty(self.bat_size_n*self.dim_y, dtype=torch.float, device=torch.device('cuda'))
		self.d_volP = torch.empty((self.bat_size_n,self.dim_y), dtype=torch.float, device=torch.device('cuda'))
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
		# d_y = cuda.as_cuda_array(self.d_G_z)
		# qrng = rand.QRNG(rndtype=rand.QRNG.SOBOL32, ndim=self.dim_y, offset=count*self.bat_size_n)
		# qrng.generate(d_y)
		# self.d_volP = self.d_G_z.view(self.dim_y,self.bat_size_n).t()

		d_z_cuda = cuda.as_cuda_array(self.d_z)
		qrng = rand.QRNG(rndtype=rand.QRNG.SOBOL32, ndim=self.dim_z, offset=count*self.bat_size_n)
		qrng.generate(d_z_cuda)
		self.d_volP = self.d_G_model(self.d_z.view(self.dim_z,self.bat_size_n).t())
		self.d_volP = self.d_volP.view(self.d_volP.shape[0],-1)

		# G_z_iter = iter(self.G_loader)
		# temp_G_z,_ = G_z_iter.next()
		# temp_G_z = temp_G_z.view(-1,self.dim_y)
		# self.d_volP.copy_(temp_G_z)
		

	def cal_measure(self):
		self.d_tot_ind_val.fill_(-1e30)
		self.d_tot_ind.fill_(-1)
		data_iter = iter(self.dataloader)
		i = 0
		while i < len(self.dataloader):
		#while i < self.numP // self.bat_size_P:
			temp_P,_ = data_iter.next()
			temp_P = temp_P.view(temp_P.shape[0],-1)	
				
			'''U=PX+H'''
			self.d_temp_h = self.d_h[i*self.bat_size_P:(i+1)*self.bat_size_P]
			self.d_temp_P.copy_(temp_P)
			#self.d_temp_P = torch.narrow(self.d_P,0,i*self.bat_size_P,self.bat_size_P)
			torch.mm(self.d_temp_P,self.d_volP.t(),out=self.d_U)
			torch.add(self.d_U,self.d_temp_h.expand([self.bat_size_n,-1]).t(),out=self.d_U)
			'''compute max'''
			torch.max(self.d_U,0,out=(self.d_ind_val, self.d_ind))
			'''add P id offset'''
			self.d_ind.add_(i*self.bat_size_P)
			'''store best value'''
			torch.max(torch.stack((self.d_tot_ind_val,self.d_ind_val)),0,out=(self.d_tot_ind_val,self.d_ind_val_argmax))
			self.d_tot_ind = torch.stack((self.d_tot_ind,self.d_ind))[self.d_ind_val_argmax, torch.arange(self.bat_size_n)] 
			'''add step'''
			i = i+1
			
		'''calculate histogram'''
		self.d_g.copy_(torch.bincount(self.d_tot_ind,minlength=self.numP))
		self.d_g.div_(self.bat_size_n)
		

	def update_h(self):
		self.d_g -= 1./self.numP
		self.d_adam_m *= 0.9
		self.d_adam_m += 0.1*self.d_g
		self.d_adam_v *= 0.999
		self.d_adam_v += 0.001*torch.mul(self.d_g,self.d_g)
		torch.mul(torch.div(self.d_adam_m, torch.add(torch.sqrt(self.d_adam_v),1e-8)),-self.lr,out=self.d_delta_h)
		torch.add(self.d_h, self.d_delta_h,out=self.d_h)
		'''normalize h'''
		self.d_h -= torch.mean(self.d_h)


	def run_gd(self):
		g_ratio = 1e20
		best_g_ratio = 1e20
		curr_best_g_ratio = 1e20
		steps = 0
		count_bad = 0
		dyn_num_bat_n = self.numP // self.bat_size_n

		while(steps <= self.maxIter):
			self.d_g_sum.fill_(0.)
			for count in range(dyn_num_bat_n):
				self.pre_cal(count)
				self.cal_measure()
				torch.add(self.d_g_sum, self.d_g, out=self.d_g_sum)
				ut.progbar(count+1,dyn_num_bat_n, 20)
			print(' ')
			torch.div(self.d_g_sum, dyn_num_bat_n, out=self.d_g)			
			self.update_h()

			g_norm = torch.sqrt(torch.sum(torch.mul(self.d_g,self.d_g)))
			num_zero = torch.sum(self.d_g == -1./self.numP)

			torch.abs(self.d_g, out=self.d_g)
			g_ratio = torch.max(self.d_g)*self.numP
			
			print('[{0}/{1}] Max absolute error ratio: {2:.3f}. g norm: {3:.6f}. num zero: {4:d}'.format(
				steps, self.maxIter, g_ratio, g_norm, num_zero))

			if g_ratio < 1e-2:
				return
			if g_ratio < best_g_ratio:
				torch.save(self.d_h, './h/{}.pt'.format(steps))
				best_g_ratio = g_ratio
			if g_ratio < curr_best_g_ratio:
				curr_best_g_ratio = g_ratio
				count_bad = 0
			else:
				count_bad += 1
			if count_bad > 20:
				dyn_num_bat_n *= 2
				print('bat_size_n has increased to {}'.format(dyn_num_bat_n*self.bat_size_n))
				count_bad = 0
				curr_best_g_ratio = 1e20

			steps += 1

			
if __name__ == '__main__':
	'''tasks'''	
	write_G_z = False
	train_omt = True

	'''args'''
	data_root = './data/sample_celebA_9000'	
	if write_G_z:
		G_z_root = './data/G_z'
	numP = 9000
	dim_y = 64*64*3
	dim_z = 100
	maxIter = 60000
	lr = 1e-1
	bat_size_P = 4500
	bat_size_n = 500

	'''model initialization'''
	g_model = Generator(dim_z).cuda()
	for param in g_model.parameters():
		param.requires_grad = False
	g_model.init_param()
	p_s = pyOMT_simple(data_root,g_model,numP,dim_y,dim_z,maxIter,lr,bat_size_P,bat_size_n)

	'''perform calculations'''
	if write_G_z:
		for count in range(2000):
			d_z = torch.empty(bat_size_n*dim_z, dtype=torch.float, device=torch.device('cuda'))
			d_z_cuda = cuda.as_cuda_array(d_z)
			qrng = rand.QRNG(rndtype=rand.QRNG.SOBOL32, ndim=dim_z, offset=count*bat_size_n)
			qrng.generate(d_z_cuda)
			d_volP = g_model(d_z.view(dim_z,bat_size_n).t())
			d_volP = d_volP.view(d_volP.shape[0],-1)
			np.savetxt(os.path.join(G_z_root,'{}_{}.gz'.format(bat_size_n, count)), d_volP.cpu().numpy())


	if train_omt:
		if not os.path.isfile('./models/initial_g_model.pth'):
			torch.save(g_model.state_dict(), './models/initial_g_model.pth')
		else:
			g_model.load_state_dict(torch.load('./models/initial_g_model.pth'))
		p_s.run_gd()
	
