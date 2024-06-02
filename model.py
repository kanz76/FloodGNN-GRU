import torch 
from torch import nn 
import torch.nn.functional as F 
import torch 
import numpy as np 
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
from layers import GVP, _norm_no_nan 
from torch_geometric.utils import add_self_loops, softmax


class FloodLayer(MessagePassing): 
    
    def __init__(self, in_dims, out_dims, activations=(F.relu, torch.sigmoid)):
        super().__init__(node_dim=0) 
        self.in_dims = in_dims 
        self.out_dims = out_dims 
        self.aggr = 'add'

        self.n_encode = GVP(self.in_dims, self.out_dims)
        self.m_gvp = GVP(self.out_dims, self.out_dims)
        self.u_gvp = GVP(self.out_dims, self.out_dims, activations=activations)
        
        
    def forward(self, edge_index, s, v):
        edge_index, _ = add_self_loops(edge_index, num_nodes=s.shape[0])
        n_nodes = s.shape[0]
        s, v = self.n_encode((s, v))
        s_out, v_out = self.propagate(edge_index, s=s, v=v, n_nodes=n_nodes)
        s_out, v_out = self.u_gvp((s_out, v_out))

        return s_out, v_out 
    
    def message(self, s_i, v_i, s_j, v_j, index, size_i):
        s_att = torch.sum(s_i * s_j, dim=1, keepdims=True)
        s_att = F.leaky_relu(s_att, negative_slope=0.2)
        s_att = softmax(s_att, index, dim=0)
        s_m_out = s_att * s_j 
        v_att = (v_i * v_j).sum(dim=(-2, -1), keepdims=True)
        v_att = F.leaky_relu(v_att, negative_slope=0.2)
        v_att = softmax(v_att, index, dim=0)
        v_m_out = v_att * v_j 
        s_m_out, v_m_out = self.m_gvp((s_m_out, v_m_out))

        return s_m_out, v_m_out

    def aggregate(self, inputs, index, n_nodes):
        s_aggr = scatter(inputs[0], index, dim=0, dim_size=n_nodes,
                reduce=self.aggr)
        v_aggr = scatter(inputs[1], index, dim=0, dim_size=n_nodes,
                reduce=self.aggr)
        
        return s_aggr, v_aggr


class LabelPred(nn.Module):
    def __init__(self, in_dims):
        super().__init__()
        self.in_dims = in_dims 
        self.gvp_layer = GVP(self.in_dims, self.in_dims)
        self.gvp_layer2 = GVP(self.in_dims, (1, 0), activations=(None, None))
        self.ln = nn.Linear(self.in_dims[0]*2, 1)

    def forward(self, s, v):
        input = (s, v)
        input = self.gvp_layer(input)
        input = self.gvp_layer2(input)
        pred = input 

        return pred 


class FeatPred(nn.Module): 
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.in_dims = in_dims 
        self.out_dims = out_dims 
        self.gvp_layer1 = GVP(self.in_dims, self.in_dims)
        self.gvp_layer2 = GVP(self.in_dims, self.out_dims, activations=(None, None))

    def forward(self, s_x, v_x):
        s_x, v_x = self.gvp_layer1((s_x, v_x))
        l_h, v_out = self.gvp_layer2((s_x, v_x))
        v_old = v_out 
        
        l_h  = l_h ** 2

        s_out = _norm_no_nan(v_out)
        v_out = v_out / s_out.unsqueeze(-1)

        return l_h, (s_out, v_out), v_old 


class FloodGNNGRU(nn.Module):  

    def __init__(self, args):
        super().__init__()
        self.static_in_dims = args.static_in_dims 
        self.in_dims = args.in_dims
        self.enc_dims = (args.s_h_dim, args.v_h_dim) 
        self.processor = FloodBlock(self.in_dims, self.enc_dims)
        self.feat_pred = FeatPred(self.enc_dims, (1, self.in_dims[1]))

    def forward(self, graphs):
        out_labels = []
        out_s_feats = []
        out_v_feats = []
        label_loss = 0
        feat_loss = 0
        edge_index = graphs.edge_index 
        s_static = graphs.s_static
        valid_seq_ind = graphs.seq
        binary = graphs.bin 
        v = torch.where(binary.bool().unsqueeze(-1), graphs.x_v , 0.0) # no-water depth, enforce no velocity.
        v_norm = torch.where(binary.bool(), graphs.x_v_norm , 0.0)
        rains = graphs.rain 
        labels = graphs.wdfp 

        s_h, v_h = v_norm[:, 0], v[:, 0]
        l_h = labels[:, 0]
        r_h = rains[: , 0]
        seq_len = labels.shape[1] - 1
        s_h_0, v_h_0 = None, None
        for i in range(1, seq_len + 1):
            s_in = torch.cat([s_static, s_h, r_h, l_h], dim=-1)
            v_in = v_h 
            b_targets = binary[:, i]
            s_h_0, v_h_0 = self.processor(edge_index, s_in, v_in, s_h_0, v_h_0, valid_seq_ind[:, i-1])

            l_h, (s_h, v_h), v_old = self.compute_regression(s_h_0, v_h_0, valid_seq_ind[:, i-1])
            
            if torch.all(b_targets  == 0) or torch.all(b_targets == 1): 
                label_loss += 0
                feat_loss += 0
            else:
                bin_mask =  b_targets.bool() 
                label_loss += self.compute_loss(labels[:, i], l_h, graphs.batch, valid_seq_ind[:, i-1])
                feat_loss += self.compute_loss(v[:, i] * v_norm[:, i].unsqueeze(-1), v_old, graphs.batch, valid_seq_ind[:, i-1]) 

            out_labels.append(l_h)
            out_s_feats.append(s_h)
            out_v_feats.append(v_h)

            r_h = rains[:, i]
        
        loss = (label_loss + feat_loss) 
        out_labels = torch.stack(out_labels, dim=1).squeeze(-1)
        out_s_feats = torch.stack(out_s_feats, dim=1)
        out_v_feats = torch.stack(out_v_feats, dim=1)
        
        out_feats = out_v_feats * out_s_feats.unsqueeze(-1) 
        out_feats = out_feats.reshape(*out_feats.shape[:-2], -1)
        assert out_feats.shape[-1] == 4 
        
        return out_labels, loss, valid_seq_ind

    def compute_regression(self, s, v, valid_seq_ind):
        
        mask = valid_seq_ind.unsqueeze(-1)
        l_h, (s_h, v_h), v_old = self.feat_pred(s, v) # Simply return the norm and normalization 
        
        l_h = torch.where(mask, l_h, 0)
        s_h = torch.where(mask, s_h, 0)
        v_h = torch.where(mask.unsqueeze(-1), v_h, 0)
        
        return l_h, (s_h, v_h), v_old


    def compute_loss(self, targets, preds, batch_index, valid_seq_ind):
        
        mask = valid_seq_ind
            
        if torch.all(~mask): 
            return 0
        targets = targets[mask]
        preds = preds[mask]
        batch_index = batch_index[mask]
        
        targets = targets.reshape(targets.shape[0], -1)
        preds = preds.reshape(targets.shape[0], -1)
        loss =  torch.abs(preds - targets)
        
        return loss.sum()


class FloodBlock(nn.Module): 
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.in_dims = in_dims 
        self.out_dims = out_dims 
        
        self.gvp_layer = GVP(self.in_dims, self.out_dims, activations=(None, None))
        
        self.z_in_conv = FloodLayer(self.out_dims, self.out_dims)
        self.z_h_conv = FloodLayer(self.out_dims, self.out_dims)
        
        self.r_in_conv = FloodLayer(self.out_dims, self.out_dims)
        self.r_h_conv = FloodLayer(self.out_dims, self.out_dims)
        
        self.h_hat_in_conv = FloodLayer(self.out_dims, self.out_dims)
        self.h_hat_h_conv = FloodLayer(self.out_dims, self.out_dims)
    
    def forward(self, g, s, v, s_hid, v_hid, valid_seq_ind):
        
        s, v = self.gvp_layer((s, v))
        mask = valid_seq_ind.unsqueeze(-1)
        
        s_g_z, v_g_z = self.z_in_conv(g, s, v)
        s_g_r, v_g_r = self.r_in_conv(g, s, v)

        if s_hid is not None:
            g_z_out = self.z_h_conv(g, s_hid, v_hid)
            s_g_z, v_g_z = s_g_z + g_z_out[0], v_g_z + g_z_out[1]
            
            g_r_out = self.r_h_conv(g, s_hid, v_hid)
            s_g_r, v_g_r = s_g_r + g_r_out[0], v_g_r + g_r_out[1]
        
        s_g_z, v_g_z= torch.sigmoid(s_g_z), torch.sigmoid(v_g_z)
        s_g_r, v_g_r= torch.sigmoid(s_g_r), torch.sigmoid(v_g_r)
        
        s_hid_hat, v_hid_hat = self.h_hat_in_conv(g, s, v)
        if s_hid is not None:
            hid_hat_out =  self.h_hat_h_conv(g, s_g_r * s_hid, v_g_r * v_hid)
            s_hid_hat, v_hid_hat = s_hid_hat + hid_hat_out[0], v_hid_hat + hid_hat_out[1]
        
        s_hid_hat, v_hid_hat = torch.tanh(s_hid_hat), torch.tanh(v_hid_hat)
        s_hid_t, v_hid_t = ( 1 - s_g_z) * s_hid_hat, (1 - v_g_z) * v_hid_hat
        
        if s_hid is not None:
           s_hid_t, v_hid_t = s_g_z * s_hid + s_hid_t, v_g_z * v_hid + v_hid_t 
        
        s_hid_t = torch.where(mask, s_hid_t, 0.0)
        v_hid_t = torch.where(mask.unsqueeze(-1), v_hid_t, 0.0)
        
        return s_hid_t, v_hid_t

