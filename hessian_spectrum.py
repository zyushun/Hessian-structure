import numpy as np
import math
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from contextlib import nullcontext
import ipdb
import json




class Hessian(object):
    def __init__(self, model = None,  m = 100, sigma = 1e-5**0.5, ckpt_iteration= 0, train_data = [], block_size = None, batch_size = None, num_v = 10, ctx =nullcontext(), use_minibatch = True, gradient_accumulation_steps = 1, loss = None, device = 'cuda',  ddp = False, comment = None):
        self.model = model
        self.m = m # number of lanzcos basis
        self.sigma = sigma # the standard deviation of gaussian r.v.
        self.ckpt_iteration = ckpt_iteration
        self.train_data = train_data
        self.block_size = block_size
        self.batch_size = batch_size
        self.ctx = ctx
        self.use_minibatch = use_minibatch
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = device
        self.ddp = ddp
        self.num_v = num_v
        self.loss = loss
        self.num_bins = 1000


        self.num_batches = len(self.train_data)
        
        #print('total batch', self.num_batches)

        self.total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        #n_params = sum(p.numel() for p in self.parameters())
        #print('total params', self.total_params)
        #time.sleep(3)
        

        self.comment = comment

        self.file_dir = 'files/'+str(self.comment)+'/'

        os.makedirs(self.file_dir, exist_ok= True)


    def get_spectrum(self, layer_by_layer = False):
        if layer_by_layer: 
            self.get_spectrum_layer_by_layer()
        else: 
            self.get_spectrum_full()


    def get_spectrum_layer_by_layer(self):
      
        weights_dic, values_dic = {}, {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                weights_dic[name] = np.zeros((self.num_v, self.m))
                values_dic[name] = np.zeros((self.num_v, self.m))
               
        t_s = time.time()
        for k in range(self.num_v): 
            'wiki version'
            T_dic = self.tridiagonalize_by_lanzcos_layer_by_layer(k) #returns a dic: {'name': T}
            
            for name, T in T_dic.items():
                eigenvalues, U  = np.linalg.eigh(T)
                values_dic[name][k,:] = eigenvalues
                weights_dic[name][k,:] = U[0]**2

            'we also save the inter-medium results'
            self.save_curve(total_time= time.time() - t_s, weights_layer = weights_dic, values_layer = values_dic)

        total_time = time.time() - t_s

        self.save_curve(total_time= total_time, weights_layer = weights_dic, values_layer = values_dic)



    def get_spectrum_full(self):
      
        weights = np.zeros((self.num_v, self.m))
        values = np.zeros((self.num_v, self.m))
        time_initial = time.time()

        for k in range(self.num_v): 
            'wiki version'
            T = self.tridiagonalize_by_lanzcos(k)
            eigenvalues, U  = np.linalg.eigh(T)
            values[k,:] = eigenvalues
            weights[k,:] = U[0]**2
   

            self.save_curve(total_time = time.time() -time_initial, weights_full =  {'weights': weights}, values_full = {'values': values}, grid = [], curve = [])
            
        total_time = time.time() -time_initial
        grid, curve = self.interpolate(weights, values)

        self.save_curve(total_time = total_time, weights_full =  {'weights': weights}, values_full = {'values': values}, grid = grid, curve = curve)

    def save_curve(self,total_time = None, weights_layer = None, values_layer = None, weights_full = None, values_full = None, grid = [], curve = []):

        if total_time != None:         
            file_name = self.file_dir + 'time.txt'
            with open(file_name, "w") as file:
                file.write(str(total_time) + "\n")

        if weights_layer != None:
            weights_layer = {key: weights_layer[key].tolist() for key in weights_layer} # convert the values to list
            file_name = self.file_dir + 'weights_layer.json'
            with open(file_name, 'w') as json_file:
                json.dump(weights_layer, json_file)
        

        if values_layer != None:
            values_layer = {key: values_layer[key].tolist() for key in values_layer} # convert the values to list
            file_name = self.file_dir + 'values_layer.json'
            with open(file_name, 'w') as json_file:
                json.dump(values_layer, json_file)

        if weights_full != None:
            weights_full = {key: weights_full[key].tolist() for key in weights_full} # convert the values to list
            file_name = self.file_dir + 'weights_full.json'
            with open(file_name, 'w') as json_file:
                json.dump(weights_full, json_file)
        

        if values_full != None:
            values_full = {key: values_full[key].tolist() for key in values_full} # convert the values to list
            file_name = self.file_dir + 'values_full.json'
            with open(file_name, 'w') as json_file:
                json.dump(values_full, json_file)


        if len(grid) != 0: 
            file_name = self.file_dir+ 'grid.txt'
            with open(file_name, "w") as file:
                for item in grid:
                    file.write(str(item) + "\n")

        if len(curve) != 0:
            file_name =  self.file_dir + 'curve.txt'
            with open(file_name, "w") as file:
                for item in curve:
                    file.write(str(item) + "\n")
        

    def load_curve(self, layer_by_layer = False, plot_histogram = False):
        if layer_by_layer: 
            self.load_curve_layer_by_layer(plot_histogram = plot_histogram)
        else: 
            self.load_curve_full(plot_histogram = plot_histogram)

      
    def load_curve_layer_by_layer(self, plot_histogram = False):

        'load weights and values:'
        file_name = self.file_dir + 'weights_layer.json'
        with open(file_name, 'r') as json_file:
            weights_dic = json.load(json_file)
        weights_dic = {key: np.array(value) for key, value in weights_dic.items()}


        file_name = self.file_dir + 'values_layer.json'
        with open(file_name, 'r') as json_file:
            values_dic = json.load(json_file)
        values_dic = {key: np.array(value) for key, value in values_dic.items()}


        if plot_histogram: 
            'load true eigen value' 
            file_name = self.file_dir + 'eigenvalues_layer.json'
            
            with open(file_name, 'r') as json_file:
                eigenvalues_dic = json.load(json_file)
            # eigenvalues_dic = {key: np.array(eigen) for key, eigen in eigenvalues_dic.items()}
            param_dict = {}
            for name, param in self.model.named_parameters():
                param_dict[name] = torch.flatten(param)



        for name in weights_dic.keys():
            weights = weights_dic[name]
            values = values_dic[name]
            grid, curve = self.interpolate(weights, values)

            print('curve',curve)

            if plot_histogram:

                eigenvalues = eigenvalues_dic[name]
                true_curve = self.get_true_curve(grid, np.array(eigenvalues))
                total_error = np.sum(np.abs(np.array(true_curve) - np.array(curve)))*(grid[1] - grid[0])
                total_para = param_dict[name].size(0)

                print('name=', name, 'total_param', total_para, 'total_error', total_error, 'grid size', grid[1]- grid[0] )

            'plot'
            plt.figure()

            if plot_histogram:
                bins = np.linspace(np.min(eigenvalues), np.max(eigenvalues), self.num_bins).tolist()
                plt.hist(eigenvalues,  bins=bins, density = True, edgecolor='black', label = 'True spectrum', alpha = 1, log = False)
                plt.plot(grid, true_curve, label = 'true curve')

            plt.plot(grid, curve, label = 'approximated curve', alpha = 0.5)
            plt.xlabel('Eigenvalues')
            plt.ylabel('Frequency')
            #plt.ylim([0,1])
            #plt.xlim([-1, 1])
            plt.legend()
            plt.title(f'model at interation {self.ckpt_iteration}')
            plt.savefig(self.file_dir+'spectrum_'+name+'.png')
            plt.close()

            'log plot'
            plt.figure()
            if plot_histogram:
                bins = np.linspace(np.min(eigenvalues), np.max(eigenvalues), self.num_bins).tolist()
                plt.hist(eigenvalues,  bins=bins, density = True, edgecolor='black', label = 'True spectrum', alpha = 1, log = True)
                plt.semilogy(grid, true_curve, label = 'true curve')
            plt.semilogy(grid, curve, label = 'approximated curve', alpha = 0.5)
            plt.xlabel('Eigenvalues')
            plt.ylabel('Frequency (log)')
            #plt.ylim([0,0.5])
            #plt.xlim([3, 5])
            plt.legend()
            plt.title(f'model at interation {self.ckpt_iteration}')
            plt.savefig(self.file_dir+'/spectrum_log_'+name+'.png')
            plt.close()


    def load_curve_full(self, plot_histogram = False):    
        'load curve'
        grid = []
        file_name = self.file_dir + 'grid.txt'
        with open(file_name, "r") as file:
            for line in file:
                grid.append(float(line.strip()))  # Use strip() to remove 

        file_name =  self.file_dir + 'curve.txt'
        curve = []
        with open(file_name, "r") as file:
            for line in file:
                curve.append(float(line.strip()))  # Use strip() to remove 


        if plot_histogram:
            'load true eigen value' 
            file_name = self.file_dir + 'eigenvalues.txt'
            eigenvalues = []
            with open(file_name, "r") as file:
                for line in file:
                    eigenvalues.append(float(line.strip()))  # Use strip() to remove 
            'get true curve'
            true_curve = self.get_true_curve(grid, np.array(eigenvalues))
            total_error = np.sum(np.abs(np.array(true_curve) - np.array(curve)))*(grid[1]- grid[0])
            # total_error = np.mean(np.abs(np.array(true_curve) - np.array(curve)))
            print('total_error', total_error, 'grid size', grid[1]- grid[0] )


        'plot'
        plt.figure()
        if plot_histogram:
            bins = np.linspace(np.min(eigenvalues), np.max(eigenvalues), self.num_bins).tolist()
            plt.hist(eigenvalues,  bins=bins, density = True, edgecolor='black', label = 'True spectrum', alpha = 1, log = False)
            plt.plot(grid, true_curve, label = 'true curve')
        plt.plot(grid, curve, label = 'approximated curve', alpha = 0.5)
        plt.xlabel('Eigenvalues')
        plt.ylabel('Frequency')
        #plt.ylim([0,1])
        # plt.xlim([-5, 5])
        plt.legend()
        plt.title(f'model at interation {self.ckpt_iteration}')
        plt.savefig(self.file_dir+'/spectrum_full_hessian.png')
        plt.close()

        'log plot'
        plt.figure()
        if plot_histogram:
            bins = np.linspace(np.min(eigenvalues), np.max(eigenvalues), self.num_bins).tolist()
            plt.hist(eigenvalues,  bins=bins, density = True, edgecolor='black', label = 'True spectrum', alpha = 1, log = True)
            true_curve = self.get_true_curve(grid, np.array(eigenvalues))
            plt.semilogy(grid, true_curve, label = 'true curve')
        plt.semilogy(grid, curve, label = 'approximated curve', alpha = 0.5)
        plt.xlabel('Eigenvalues')
        plt.ylabel('Frequency (log)')
        #plt.ylim([0,0.5])
        #plt.xlim([3, 5])
        plt.legend()
        plt.title(f'model at interation {self.ckpt_iteration}')
        plt.savefig(self.file_dir+'/spectrum_log_full_hessian.png')
        plt.close()


    def tridiagonalize_by_lanzcos_layer_by_layer(self, k):
        v_dic = {} # value: list
        alpha_dic = {} # value: scaler
        w_dic = {} # value: #parameters*1 tensor
        beta_dic = {} # value: scaler
        T_dic = {} # value: m*m tensor 
        'initialize'
        for name, params in self.model.named_parameters():
            if params.requires_grad:
                v = torch.randn_like(params, dtype = torch.float64) 
                v /= torch.norm(v)
                v_dic[name] = [v.cpu()]
                T_dic[name] = np.zeros((self.m, self.m), dtype= np.float64)


        w_prime_dic = self.hessian_vector_product_with_dic_input(v_dic, k,0) 

        'orthogonalize wprime'
        for name in T_dic.keys():
            alpha_dic[name] = torch.sum(w_prime_dic[name] * v_dic[name][-1])  
            w_dic[name] = w_prime_dic[name] - alpha_dic[name] * v_dic[name][-1]
            T_dic[name][0, 0] = alpha_dic[name] 

        'iteration'
        print('runing lanczos')
        for j in range(1, self.m):

            for name in T_dic.keys(): 
                beta = torch.norm(w_dic[name])
                beta_dic[name] = beta
                if beta >1e-8:
                    v_dic[name].append( w_dic[name] / beta )
                else:
                    #print('The value of beta is 0')
                    v_dic[name].append( w_dic[name] / 1e-8 )
                    #raise ZeroDivisionError('The value of beta is 0')
                if len(v_dic[name]) > 2:
                    del v_dic[name][0]  # keep this list short to save memory

            t_hessian = time.time()

            w_prime_dic = self.hessian_vector_product_with_dic_input(v_dic, k,j) 
            print('t for hessian', time.time() - t_hessian)

            'orthogonalize wprime'
            for name in T_dic.keys():
                alpha_dic[name] = torch.sum(w_prime_dic[name] * v_dic[name][-1])  
                w_dic[name] = w_prime_dic[name] - alpha_dic[name] * v_dic[name][-1] - beta_dic[name] * v_dic[name][-2]
                T_dic[name][j, j] = alpha_dic[name] 
                T_dic[name][j-1, j ] = beta_dic[name] 
                T_dic[name][j , j-1] = beta_dic[name]

        return  T_dic


    def tridiagonalize_by_lanzcos(self, k):
        'set up'
        v_list = []
        T = np.zeros((self.m, self.m), dtype= np.float64)

        'initialization'
        v = torch.randn(self.total_params, dtype = torch.float64) 
        v /= torch.norm(v)
        v_list.append(v.cpu())


            
        w_prime = self.hessian_vector_product_with_tensor_input(v_list[-1], k,0)
        'orthogonalize wprime'
        alpha = torch.sum(w_prime * v_list[-1])
        w = w_prime - alpha * v_list[-1]
        T[0, 0] = alpha

        'iteration'
        #t_s = time.time()
        print('runing lanczos')
        for j in range(1, self.m):
            beta = torch.norm(w)
            if beta >1e-8:
                v_list.append(w / beta)

            else:
                v_list.append(w / 1e-8)

                # print(f' since beta = {beta}, generate v that orthogonal to all previous v')
                # # Generate a random vector orthogonal to previous ones
                # v = torch.randn(self.total_params) *(1/self.total_params)**0.5
                # for i in range(j):
                #     vi = v_list[i]
                #     v -= torch.sum(vi * v) * vi
                # v /= torch.norm(v)
                if len(v_list) > 2:
                    del v_list[0]  # keep this list short to save memory

            w_prime = self.hessian_vector_product_with_tensor_input(v_list[-1], k,j)
            alpha = torch.sum(w_prime* v_list[-1])
            w = w_prime - alpha * v_list[-1] - beta * v_list[-2]
            T[j, j] = alpha
            T[j-1, j ] = beta
            T[j , j-1] = beta
            # if j % 30 == 0:
            #     print('Lanczos iteration ', j, 'time  = ', time.time()-t_s)
            #     t_s = time.time()
        return  T


    def interpolate(self,weights, values):
        left_boundary = np.mean(np.min(values, axis = 1))-1
        right_boundary= np.mean(np.max(values, axis = 1)) +1
        n_grid = 50000
        grid = np.linspace(left_boundary, right_boundary, n_grid).tolist()
        density_all = np.zeros((self.num_v, n_grid))

        for k  in range(self.num_v):
            for idx, t  in enumerate(grid):
                values_each_v_t = self.gaussian_density(t, values[k,:])
                density_each_v_t = np.sum(values_each_v_t * weights[k,:])
                density_all[k,idx] = density_each_v_t

        density_avg = np.nanmean(density_all, axis = 0)
        norm_fact = np.sum(density_avg)*(grid[1]- grid[0])
        density_avg /= norm_fact

        return grid, density_avg
 


    def hessian_vector_product_with_dic_input(self, d_dic, v_step, l_step):
        'comput hessian_vector product, takes a dictionary as input, the values of dic is a list of historical lanscoz directions: d_dic = {name, [history v..]}'
        self.model.eval()
        self.model.zero_grad(set_to_none = True)

        'initialize'
        hd_dic = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                hd_dic[name]  = torch.zeros_like(param.data).cpu()


        t_hd = time.time()
        for batch_idx, data in enumerate(self.train_data):

            X_train = data['X_train']
            Y_train = data['Y_train']

            output = self.model(X_train)

            loss = self.loss(output, Y_train) #F.cross_entropy(output, Y_train) 
            loss.backward(create_graph= True)
            
            g_dic = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    g_dic[name] = param.grad.double()
  
        
            self.model.zero_grad(set_to_none = True)
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    l = torch.sum(g_dic[name].cuda() * d_dic[name][-1].cuda())
                    l.backward(retain_graph = True)
                    hd = param.grad.double().data.clone()
                    hd_dic[name]  += hd.cpu()   
                    self.model.zero_grad(set_to_none = True)
            
        
            if batch_idx % 10 == 1 or batch_idx == self.gradient_accumulation_steps-1:
                print(f'layer hessian: load_iter ={self.ckpt_iteration}, current random direction = {v_step} / {self.num_v}, lanczos step = {l_step} / {self.m}, Hd current batch = {batch_idx} / {self.num_batches}, time = {time.time() -t_hd}')
                t_hd = time.time()

            if self.use_minibatch == True and batch_idx == self.gradient_accumulation_steps-1:
                break

    
        # could change to list if like
        #hd_list = list(hd_dic.values())
        return hd_dic



    def hessian_vector_product_with_tensor_input(self, d_tensor, v_step, l_step):
        'comput hessian_vector product, takes a flattened tensors as input (with shape (total parameters, ) )'

        d_tensor = d_tensor.cuda()
        self.model.eval()
        self.model.zero_grad(set_to_none = True)
        total_hd_tensor = 0

        t_hd = time.time()
        for batch_idx, data in enumerate(self.train_data):

            X_train = data['X_train']
            Y_train = data['Y_train']

            output = self.model(X_train)

            loss = self.loss(output, Y_train) #F.cross_entropy(output, Y_train) 

            loss.backward(create_graph= True)
            g_list = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    g_list.append(torch.flatten(param.grad.double()))

            g_tensor = torch.cat(g_list, dim = 0)
            
            self.model.zero_grad(set_to_none = True)
            g_tensor = g_tensor.cuda()
            l = torch.sum(g_tensor*d_tensor)
            l.backward(retain_graph = True)

            hd_list = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    hd_list.append(torch.flatten(param.grad.double().data.clone()))

            hd_tensor = torch.cat(hd_list, dim = 0)
            self.model.zero_grad(set_to_none = True)
            hd_tensor = hd_tensor.cpu()
            total_hd_tensor += hd_tensor

            if batch_idx % 10 == 1 or batch_idx == self.gradient_accumulation_steps-1:
                print(f'full hessian: load_iter ={self.ckpt_iteration} current random direction = {v_step} / {self.num_v}, lanczos step = {l_step} / {self.m}, Hd current batch = {batch_idx} / {self.num_batches}, time = {time.time() -t_hd}')
                t_hd = time.time()

            if self.use_minibatch == True and batch_idx == self.gradient_accumulation_steps-1:
                break
        return total_hd_tensor


    def get_full_hessian(self):
        self.model.eval()
        self.model.zero_grad(set_to_none = True)

        def hessian_calculation(g_tensor, batch_idx):
            g_tensor = g_tensor.cuda()
            total_params = g_tensor.size(0)
            hessian_list = []
            t_d = time.time()
            for d in range(total_params):
                unit_vector = torch.zeros(total_params)
                unit_vector[d] = 1
                unit_vector = unit_vector.cuda()
                l = torch.sum(g_tensor * unit_vector)
                l.backward(retain_graph= True)

                hessian_row = []
                for name, param in self.model.named_parameters():
                    if 'ln' in name or 'bias' in name or 'wte' in name or 'wpe' in name:
                        continue
                    if param.requires_grad:
                        #print('name',name, param.grad)
                        hessian_row.append(param.grad.double().data.clone())
                
                self.model.zero_grad(set_to_none = True)
                hessian_row = [g.flatten() for g in hessian_row] 
                hessian_row = [g.cpu() for g in hessian_row]
                hessian_row = torch.cat(hessian_row)
                #print('hessian_row', hessian_row)   
                hessian_list.append(hessian_row)
                # if d % 1000 == 0:
                #     print(f'Computing hessian: current batch = {batch_idx}/{self.num_batches}, current row of a hessian: {d}/{total_params}, total time = {time.time()- t_d} ')

            hessian = torch.stack(hessian_list, dim = 1)

            #print('hessian', hessian)   
            return hessian



        full_hessian = 0

        for batch_idx, data in enumerate(self.train_data):

            X_train = data['X_train']
            Y_train = data['Y_train']

            output = self.model(X_train)

            #loss = F.cross_entropy(output, Y_train) 
            loss =self.loss(output, Y_train)

            loss.backward(create_graph= True)

            g_list = []
            count = 0
            for name, param in self.model.named_parameters():
                #if 'ln' in name or 'bias' in name:
                if 'ln' in name or 'bias' in name or 'wte' in name or 'wpe' in name:
                    continue
                if param.requires_grad:
                    count += param.numel()
                    #print('g shape', param.grad , param.grad.shape)
                    g_list.append(torch.flatten(param.grad.double()))
                    #print('name',name, g_list[-1].size())

            g_tensor = torch.cat(g_list, dim = 0)
            #print('g_tensor',g_tensor)
            self.model.zero_grad(set_to_none = True)
            H = hessian_calculation(g_tensor, batch_idx)
            full_hessian += H


        full_hessian = torch.nan_to_num(full_hessian, nan = 0, posinf = 0, neginf = 0 )  # change nan, postive inf , negative inf, to 0
        t_svd = time.time()
        #print('doing EVD')
        # _, eigenvalues, _ = torch.linalg.svd(full_hessian)  # ascending
        #eigenvalues, _  = torch.eig(full_hessian)
        full_hessian = full_hessian.numpy().astype(np.float64)
        full_hessian = (full_hessian + full_hessian.T)/2 # make symetric, to 
        
        
        
        #avoid numerical issue
        #full_hessian = full_hessian.cuda()
        #eigenvalues, _  = torch.linalg.eig(full_hessian)
        # eigenvalues, _  = np.linalg.eigh(full_hessian)
        # #_, eigenvalues, _ = np.linalg.svd(full_hessian) 
        # eigenvalues = [eigen.item().real for eigen in eigenvalues]

        # file_name = self.file_dir + 'eigenvalues.txt'
        # with open(file_name, "w") as file:
        #     for item in eigenvalues:
        #         file.write(str(item)+"\n")

        # print(f'EVD time = {time.time()- t_svd}')

        return full_hessian


    def get_full_hessian_layer_by_layer(self):
        self.model.eval()
        self.model.zero_grad(set_to_none = True)

        def hessian_calculation(g_name, g_tensor, batch_idx):
            g_tensor = g_tensor.cuda()
            total_params = g_tensor.size(0)
            hessian_list = []
            t_d = time.time()
            for d in range(total_params):
                unit_vector = torch.zeros(total_params)
                unit_vector[d] = 1
                unit_vector = unit_vector.cuda()
                l = torch.sum(g_tensor*unit_vector)
                l.backward(retain_graph= True)

                hessian_row = []
                for name, param in self.model.named_parameters():
                    if name == g_name:
                        hessian_row.append(param.grad.double().data.clone())
                
                self.model.zero_grad(set_to_none = True)
                hessian_row = [g.flatten() for g in hessian_row] 
                hessian_row = [g.cpu() for g in hessian_row]
                hessian_row = torch.cat(hessian_row)
                hessian_list.append(hessian_row)
                # if d % 1000 == 0:
                #     print(f'Computing hessian: current batch = {batch_idx}/{self.num_batches}, current row of a hessian: {d}/{total_params}, total time = {time.time()- t_d} ')

            hessian = torch.stack(hessian_list, dim = 1)
            return hessian



        full_hessian_dic = {}
        'initialization'
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                #size = torch.flatten(param.data).size(0)
                full_hessian_dic[name] = 0 #torch.zeros(size, size)

        for batch_idx, data in enumerate(self.train_data):

            X_train = data['X_train']
            Y_train = data['Y_train']

            output = self.model(X_train)

            loss = self.loss(output, Y_train) #F.cross_entropy(output, Y_train) 
            loss.backward(create_graph= True)

            g_dic = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    g_dic[name] = torch.flatten(param.grad.double())

            #g_tensor = torch.cat(g_list, dim = 0)
            self.model.zero_grad(set_to_none = True)

            for name, g_tensor in g_dic.items():
                H = hessian_calculation(name, g_tensor, batch_idx)
                H = torch.nan_to_num(H, nan = 0, posinf = 0, neginf = 0 )  # change nan, postive inf , negative inf, to 0
                H = H.numpy().astype(np.float64)
                H = (H + H.T)/2
                full_hessian_dic[name] = H

        return full_hessian_dic
        # t_svd = time.time()
        # eigenvalues_dic = {}
        # for name, full_hessian in full_hessian_dic.items():
        #     full_hessian = torch.nan_to_num(full_hessian, nan = 0, posinf = 0, neginf = 0 )  # change nan, postive inf , negative inf, to 0
        #     # _, eigenvalues, _ = torch.linalg.svd(full_hessian)  # ascending
        #     #eigenvalues, _  = torch.eig(full_hessian)
        #     full_hessian = full_hessian.numpy().astype(np.float64)
        #     full_hessian = (full_hessian + full_hessian.T)/2 # make symetric, to avoid numerical issue
        #     #full_hessian = full_hessian.cuda()
        #     #eigenvalues, _  = torch.linalg.eig(full_hessian)
        #     eigenvalues, _  = np.linalg.eigh(full_hessian)
        #     #_, eigenvalues, _ = np.linalg.svd(full_hessian) 
        #     eigenvalues = [eigen.item().real for eigen in eigenvalues]
        #     eigenvalues_dic[name] = eigenvalues

        # file_name = self.file_dir + 'eigenvalues_layer.json'
        # with open(file_name, 'w') as json_file:
        #     json.dump(eigenvalues_dic, json_file)

        # print(f'EVD time = {time.time()- t_svd}')





