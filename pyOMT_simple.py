import sys

import torch
from torch.utils.data import DataLoader
from pyculib import rand
from numba import cuda
import P_loader
import pdb

# from models_64x64 import Generator
# from AE_gen import AE_gen
from dcgan import DCGAN_G

import torchvision
from torchvision import transforms
import pyOMT_utils as ut
import numpy as np
import os
from glob import glob
from torch import nn
import torch.optim as optim
from PIL import Image

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
		self.dataset = P_loader.P_loader(root=input_P,transform=torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
			]))
		self.dataloader = DataLoader(self.dataset, batch_size=bat_size_P, shuffle=False, pin_memory=True, drop_last=True, num_workers = 7)
		# self.G_set = P_loader.P_loader(root='./data/G_z',loader=P_loader.G_z_loader)
		# self.G_loader = DataLoader(self.G_set, batch_size=bat_size_n//500, shuffle=False, drop_last=True, num_workers = 8)
		self.d_G_model = d_G_model
		self.numP = numP
		self.dim_z = dim_z
		self.dim_y = dim_y
		self.maxIter= maxIter
		self.lr = lr
		self.bat_size_P = bat_size_P
		self.bat_size_n = bat_size_n

		# if numP % bat_size_P != 0:
		# 	sys.exit('Error: (numP) is not a multiple of (bat_size_P)')

		# if bat_size_n % 500 != 0:
		# 	sys.exit('Error: (bat_size_n) must be a multiple of 500')
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
		# pdb.set_trace()

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


	def run_gd(self, last_step=0):
		g_ratio = 1e20
		best_g_norm = 1e20
		curr_best_g_ratio = 1e20
		steps = 0
		count_bad = 0
		dyn_num_bat_n = 6
		# pdb.set_trace()
		h_file_list = []
		m_file_list = []
		v_file_list = []

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

			if g_ratio < 1e-2 or num_zero <= self.numP/2:
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

			if g_ratio <= curr_best_g_ratio:
				curr_best_g_ratio = g_ratio
				count_bad = 0
			else:
				count_bad += 1
			if count_bad > 8:
				dyn_num_bat_n *= 2
				print('bat_size_n has increased to {}'.format(dyn_num_bat_n*self.bat_size_n))
				count_bad = 0
				curr_best_g_ratio = 1e20

			steps += 1


	def set_h(self, h_tensor):
		self.d_h.copy_(h_tensor)

	def set_adam_m(self, adam_m_tensor):
		self.d_adam_m.copy_(adam_m_tensor)

	def set_adam_v(self, adam_v_tensor):
		self.d_adam_v.copy_(adam_v_tensor)

	def T_map(self, x):
		numX = x.shape[0]
		x = x.view(numX,-1)
		result_id = torch.empty([numX], dtype=torch.long, device=torch.device('cuda'))
		result = torch.empty([numX, dim_y], dtype=torch.float, device=torch.device('cuda'))
		for ii in range(numX//500 + 1):			
			x_bat = x[ii*500 : min((ii+1)*500, numX)]
			tot_ind_val = torch.empty([x_bat.shape[0]],dtype=torch.float, device=torch.device('cuda'))
			tot_ind_val.fill_(-1e30)
			tot_ind = torch.empty([x_bat.shape[0]],dtype=torch.long, device=torch.device('cuda'))
			tot_ind.fill_(-1)
			ind_val_argmax = torch.empty([x_bat.shape[0]],dtype=torch.long, device=torch.device('cuda'))
			ind_val_argmax.fill_(-1)

			data_iter = iter(self.dataloader)
			i = 0
			while i < len(self.dataloader):
				temp_P,_ = data_iter.next()
				temp_P = temp_P.view(temp_P.shape[0],-1)	
				
				'''U=PX+H'''
				self.d_temp_h = self.d_h[i*self.bat_size_P:(i+1)*self.bat_size_P]
				self.d_temp_P.copy_(temp_P)
				U = torch.mm(self.d_temp_P,x_bat.t())
				U = torch.add(U,self.d_temp_h.expand([x_bat.shape[0],-1]).t())
				'''compute max'''
				ind_val, ind = torch.max(U,0)
				curr_result = self.d_temp_P[ind]

				ind.add_(i*self.bat_size_P)
				
				torch.max(torch.stack((tot_ind_val,ind_val)),0,out=(tot_ind_val,ind_val_argmax))
				tot_ind = torch.stack((tot_ind,ind))[ind_val_argmax, torch.arange(x_bat.shape[0])] 

				
				result[ii*500 : min((ii+1)*500, numX)] = torch.cat(
					(result[ii*500 : min((ii+1)*500, numX)],curr_result), dim=0)[
					ind_val_argmax * x_bat.shape[0] + torch.arange(x_bat.shape[0]).cuda()]
				i+=1
			result_id[ii*500 : min((ii+1)*500, numX)] = tot_ind
		return result, result_id




def load_last_model(model):
    models = glob('./models/*.pth')
    model_ids = [(int(f.split('_')[1]), f) for f in models]
    if not model_ids:
        start_epoch = 0
        last_cp = 0
        return start_epoch, last_cp
    else:
        start_epoch, last_cp = max(model_ids, key=lambda item:item[0])
        print('Last checkpoint: ', last_cp)
        model.load_state_dict(torch.load(last_cp))
        return start_epoch, last_cp

def load_last_file(path, file_ext):
	files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
	file_ids = [(int(f.split('.')[0]), os.path.join(path,f)) for f in files]
	if not file_ids:
		return None, None
	else:
		last_f_id, last_f = max(file_ids, key=lambda item:item[0])
		print('Last' + path + ': ', last_f_id)
		return last_f_id, last_f

def train_omt():
	last_step = 0
	'''freeze model parameters'''
	for param in g_model.parameters():
		param.requires_grad = False
	g_model.eval()

	'''load last trained model parameters and last omt parameters'''
	load_last_model(g_model)
	h_id, h_file = load_last_file('./h', '.pt')
	adam_m_id, m_file = load_last_file('./adam_m', '.pt')
	adam_v_id, v_file = load_last_file('./adam_v', '.pt')
	if h_id != None:
		if h_id != adam_m_id or h_id!= adam_v_id:
			sys.exit('Error: h, adam_m, adam_v file log does not match')
		else:
			last_step = h_id
			p_s.set_h(torch.load(h_file))
			p_s.set_adam_m(torch.load(m_file))
			p_s.set_adam_v(torch.load(v_file))

	'''run gradient descent'''
	p_s.run_gd(last_step=last_step)


def train_G():
	'''load model'''
	start_epoch, _ = load_last_model(g_model)

	'''list of saved models'''
	list_model = []
	'''unfreeze model parameters'''
	for param in g_model.parameters():
		param.requires_grad = True

	'''load h'''
	last_h_id, last_h = load_last_file('./h','.pt')
	# pdb.set_trace()
	if last_h_id != None:
		p_s.set_h(torch.load(last_h))	
	else:
		sys.exit('Error: Cannot find h file on training G')

	'''set loss funciton'''
	optimizer = optim.Adam(g_model.parameters(), lr=train_lr, betas=(0.5, 0.999))

	'''compute target P in prior'''	
	tot_p_files = [x[0] for x in p_s.dataset.imgs]
	tot_p_id = torch.empty([train_num_z], dtype=torch.long, device=torch.device('cuda'))
	print('Preparing P...')		
	g_model.eval()
	for ii in range(train_num_z//train_bat_size):
		'''generate z'''
		z = torch.empty([train_bat_size*dim_z], dtype=torch.float, device=torch.device('cuda'))
		d_z_cuda = cuda.as_cuda_array(z)
		qrng = rand.QRNG(rndtype=rand.QRNG.SOBOL32, ndim=dim_z, offset=ii*train_bat_size)
		qrng.generate(d_z_cuda)
		'''compute y'''
		z.requires_grad_()
		y = g_model(z.view(dim_z,train_bat_size).t())
		y = y.view(y.shape[0],-1)
		'''get corresponding p'''
		p, p_id = p_s.T_map(y.detach())
		tot_p_id[ii*train_bat_size : ii*train_bat_size+y.shape[0]] = p_id
		ut.progbar(ii+1,train_num_z//train_bat_size, 20)
	print('')

	'''training'''
	sample_y = torch.empty([train_bat_size, dim_y], dtype=torch.float, device=torch.device('cuda'))
	sample_p = torch.empty([train_bat_size, dim_y], dtype=torch.float, device=torch.device('cuda'))
	
	for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
		set_sample = False
		g_model.train()
		train_loss = 0
		rand_perm = np.random.permutation(train_num_z//train_bat_size)
		ii = 0
		while ii < train_num_z//train_bat_size:
			'''generate z'''
			z = torch.empty([train_bat_size*dim_z], dtype=torch.float, device=torch.device('cuda'))
			d_z_cuda = cuda.as_cuda_array(z)
			qrng = rand.QRNG(rndtype=rand.QRNG.SOBOL32, ndim=dim_z, offset=rand_perm[ii]*train_bat_size)				
			qrng.generate(d_z_cuda)

			'''zero parameter grads'''
			optimizer.zero_grad()

			'''compute y'''
			# z.requires_grad_()
			y = g_model(z.view(dim_z,train_bat_size).t())
			y = y.view(y.shape[0],-1)
			
			if set_sample:
				sample_y.copy_(y)

			'''get corresponding p'''				
			start = torch.tensor(rand_perm[ii].item()*train_bat_size, dtype=torch.long, device=torch.device('cuda'))
			length = torch.tensor(train_bat_size, dtype=torch.long, device=torch.device('cuda'))
			# p_id = tot_p_id[rand_perm[ii].item()*train_bat_size:(rand_perm[ii].item()+1)*train_bat_size]				
			p_id = torch.narrow(tot_p_id, 0, start, length)
			p_files = [tot_p_files[ind] for ind in p_id]
			p = torch.empty([y.shape[0],dim_y], dtype=torch.float, device=torch.device('cuda'))
			for i in range(len(p_files)):
				p[i] = (transforms.ToTensor()(Image.open(p_files[i]))).view(1,-1).cuda()
			if not set_sample:
				sample_p.copy_(p)
				set_sample=True

			'''compute loss'''				
			# loss = torch.sum(torch.sum(-torch.mm(p,y.t()),dim=1))
			loss = torch.nn.MSELoss()(y,p)
			# loss = nn.BCEWithLogitsLoss()(y,p)

			'''back propogate'''				
			loss.backward()
			train_loss += loss.cpu().data.numpy()
			optimizer.step()
			'''show progress'''				
			print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
            epoch, ii, train_num_z//train_bat_size,
            100. * ii / (train_num_z//train_bat_size), loss.cpu().data / train_bat_size))

			ii+= 1
				
		'''print'''
		print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / train_num_z))
		'''save result'''
		torch.save(g_model.state_dict(), './models/Epoch_{}_Train_loss_{:.4f}.pth'.format(epoch, train_loss))
		list_model.append('./models/Epoch_{}_Train_loss_{:.4f}.pth'.format(epoch, train_loss))
		while len(list_model) > 5:
			if os.path.exists(list_model[0]):
				os.remove(list_model[0])
			list_model.pop(0)


		torchvision.utils.save_image(
			sample_p.view(sample_p.shape[0], im_c, im_h, im_w).cpu().data,
			 './imgs/input.png', nrow = 8)

		g_model.eval()
		torchvision.utils.save_image(
			sample_y.view(sample_y.shape[0], im_c, im_h, im_w).cpu().data,
			 './imgs/output_{}.png'.format(epoch), nrow = 8)

		
		

if __name__ == '__main__':
	'''tasks'''	
	# if_write_G_z = False
	if_train_omt = False
	if_train_G = False
	if_train_both = True


	'''args for omt'''
	data_root = './data/CelebA_crop_resize_64'	
	# if if_write_G_z:
	# 	G_z_root = './data/G_z'
	numP = 36679
	im_w = 64
	im_h = 64
	im_c = 3
	dim_y = im_c*im_h*im_w
	dim_z = 100
	maxIter = 60000
	lr = 5e-1
	bat_size_P = 5321
	bat_size_n = 2000


	'''args for training G model'''
	train_bat_size = 64
	train_num_z = 10000
	train_lr = 2e-5
	epochs = 1000


	'''model initialization'''
	# g_model = Generator(dim_z).cuda()	
	# g_model = AE_gen(3, 128, 128, 100).cuda()
	g_model = DCGAN_G(64,100,3,64,1).cuda()

	p_s = pyOMT_simple(data_root,g_model,numP,dim_y,dim_z,maxIter,lr,bat_size_P,bat_size_n)

	if len(os.listdir('./models')) == 0:
		# pdb.set_trace()
		# g_model.init_param()
		g_model.apply(ut.weights_init)
		torch.save(g_model.state_dict(), './models/Epoch_0_initial_g_model.pth')

	'''perform calculations'''
	# if if_write_G_z:
	# 	for count in range(2000):
	# 		d_z = torch.empty(bat_size_n*dim_z, dtype=torch.float, device=torch.device('cuda'))
	# 		d_z_cuda = cuda.as_cuda_array(d_z)
	# 		qrng = rand.QRNG(rndtype=rand.QRNG.SOBOL32, ndim=dim_z, offset=count*bat_size_n)
	# 		qrng.generate(d_z_cuda)
	# 		d_volP = g_model(d_z.view(dim_z,bat_size_n).t())
	# 		d_volP = d_volP.view(d_volP.shape[0],-1)
	# 		np.savetxt(os.path.join(G_z_root,'{}_{}.gz'.format(bat_size_n, count)), d_volP.cpu().numpy())


	if if_train_omt:
		train_omt()
	
	if if_train_G:
		train_G()

	if if_train_both:
		# count = 0
		# while True:
		# 	if count == 0:
		# 		train_G()
		# 	else:
		# 		# pdb.set_trace()
		# 		train_omt()
		# 		train_G()
		# 	count += 1
		while True:
			train_omt()
			train_G()

			