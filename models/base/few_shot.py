import torch
from torch.functional import norm
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from collections import OrderedDict
import math
from itertools import combinations
from torch.nn.init import xavier_normal_ 
import numpy as np
# from torch.nn.modules.activation import MultiheadAttention

from torch.autograd import Variable
import torchvision.models as models
# from utils import extract_class_indices, cos_sim
from einops import rearrange
import os
from torch.autograd import Variable

from utils.registry import Registry
from models.base.backbone import BACKBONE_REGISTRY
from models.base.base_blocks import HEAD_REGISTRY


# MODEL_REGISTRY = Registry("Model")
# STEM_REGISTRY = Registry("Stem")
# BRANCH_REGISTRY = Registry("Branch")
# HEAD_REGISTRY = Registry("Head")
# HEAD_BACKBONE_REGISTRY = Registry("HeadBackbone")

class PreNormattention(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) + x


class PostNormattention(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs) + x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


def cos_sim(x, y, epsilon=0.01):
    """
    Calculates the cosine similarity between the last dimension of two tensors.
    """
    numerator = torch.matmul(x, y.transpose(-1,-2))
    xnorm = torch.norm(x, dim=-1).unsqueeze(-1)
    ynorm = torch.norm(y, dim=-1).unsqueeze(-1)
    denominator = torch.matmul(xnorm, ynorm.transpose(-1,-2)) + epsilon
    dists = torch.div(numerator, denominator)
    return dists


def extract_class_indices(labels, which_class):
    """
    Helper method to extract the indices of elements which have the specified label.
    :param labels: (torch.tensor) Labels of the context set.
    :param which_class: Label for which indices are extracted.
    :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
    """
    class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
    class_mask_indices = torch.nonzero(class_mask, as_tuple=False)  # indices of labels equal to which class
    return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector



class CNN_FSHead(nn.Module):
    """
    Base class which handles a few-shot method. Contains a resnet backbone which computes features.
    """
    def __init__(self, cfg):
        super(CNN_FSHead, self).__init__()
        args = cfg
        self.train()
        self.args = args

        last_layer_idx = -1
        
        if self.args.VIDEO.HEAD.BACKBONE_NAME == "resnet18":
            backbone = models.resnet18(pretrained=True) 

        elif self.args.VIDEO.HEAD.BACKBONE_NAME == "resnet34":
            backbone = models.resnet34(pretrained=True)

        elif self.args.VIDEO.HEAD.BACKBONE_NAME == "resnet50":
            backbone = models.resnet50(pretrained=True)

        self.backbone = nn.Sequential(*list(backbone.children())[:last_layer_idx])

    def get_feats(self, support_images, target_images):
        """
        Takes in images from the support set and query video and returns CNN features.
        """
        support_features = self.backbone(support_images).squeeze()
        target_features = self.backbone(target_images).squeeze()

        dim = int(support_features.shape[1])

        support_features = support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)
        target_features = target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)

        return support_features, target_features

    def forward(self, support_images, support_labels, target_images):
        """
        Should return a dict containing logits which are required for computing accuracy. Dict can also contain
        other info needed to compute the loss. E.g. inter class distances.
        """
        raise NotImplementedError

    def distribute_model(self):
        """
        Use to split the backbone evenly over all GPUs. Modify if you have other components
        """
        if self.args.TRAIN.DDP_GPU > 1:
            self.backbone.cuda(0)
            self.backbone = torch.nn.DataParallel(self.backbone, device_ids=[i for i in range(0, self.args.TRAIN.DDP_GPU)])
    
    def loss(self, task_dict, model_dict):
        """
        Takes in a the task dict containing labels etc.
        Takes in the model output dict, which contains "logits", as well as any other info needed to compute the loss.
        Default is cross entropy loss.
        """
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())
        
        


class PositionalEncoding(nn.Module):
    """
    Positional encoding from the Transformer paper.
    """
    def __init__(self, d_model, dropout, max_len=5000, pe_scale_factor=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_scale_factor = pe_scale_factor
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) * self.pe_scale_factor
        pe[:, 1::2] = torch.cos(position * div_term) * self.pe_scale_factor
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
                          
    def forward(self, x):
       x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
       return self.dropout(x)


@HEAD_REGISTRY.register()
class TemporalCrossTransformer(nn.Module):
    """
    A temporal cross transformer for a single tuple cardinality. E.g. pairs or triples.
    """
    def __init__(self, cfg, temporal_set_size=3):
        super(TemporalCrossTransformer, self).__init__()
        # temporal_set_size=3
        args = cfg
        
        self.args = args
        if self.args.VIDEO.HEAD.BACKBONE_NAME == "resnet50":
            self.trans_linear_in_dim = 2048
        else:
            self.trans_linear_in_dim = 512
        # self.trans_linear_in_dim = 2048
        self.temporal_set_size = temporal_set_size

        max_len = int(self.args.DATA.NUM_INPUT_FRAMES * 1.5)
        self.pe = PositionalEncoding(self.trans_linear_in_dim, self.args.trans_dropout, max_len=max_len)

        self.k_linear = nn.Linear(self.trans_linear_in_dim * temporal_set_size, self.args.trans_linear_out_dim)#.cuda()
        self.v_linear = nn.Linear(self.trans_linear_in_dim * temporal_set_size, self.args.trans_linear_out_dim)#.cuda()

        self.norm_k = nn.LayerNorm(self.args.trans_linear_out_dim)
        self.norm_v = nn.LayerNorm(self.args.trans_linear_out_dim)
        
        self.class_softmax = torch.nn.Softmax(dim=1)
        
        # generate all tuples
        frame_idxs = [i for i in range(self.args.DATA.NUM_INPUT_FRAMES)]
        frame_combinations = combinations(frame_idxs, temporal_set_size)
        self.tuples = nn.ParameterList([nn.Parameter(torch.tensor(comb), requires_grad=False) for comb in frame_combinations])
        self.tuples_len = len(self.tuples) 
    
    
    def forward(self, support_set, support_labels, queries):
        n_queries = queries.shape[0]   # [35, 8, 2048]
        n_support = support_set.shape[0]   # [5, 8, 2048]
        
        # static pe
        support_set = self.pe(support_set)   # [5, 8, 2048]
        queries = self.pe(queries)      # queries

        # construct new queries and support set made of tuples of images after pe
        s = [torch.index_select(support_set, -2, p).reshape(n_support, -1) for p in self.tuples]
        q = [torch.index_select(queries, -2, p).reshape(n_queries, -1) for p in self.tuples]
        support_set = torch.stack(s, dim=-2)   # [5, 28, 4096]
        queries = torch.stack(q, dim=-2)    # [35, 28, 4096]

        # apply linear maps
        support_set_ks = self.k_linear(support_set)   # [5, 28, 1152]
        queries_ks = self.k_linear(queries)           # [35, 28, 1152]
        support_set_vs = self.v_linear(support_set)   # [5, 28, 1152]
        queries_vs = self.v_linear(queries)           # [35, 28, 1152]
        
        # apply norms where necessary
        mh_support_set_ks = self.norm_k(support_set_ks)
        mh_queries_ks = self.norm_k(queries_ks)
        mh_support_set_vs = support_set_vs
        mh_queries_vs = queries_vs
        
        unique_labels = torch.unique(support_labels)   # [0., 1., 2., 3., 4.]

        # init tensor to hold distances between every support tuple and every target tuple
        all_distances_tensor = torch.zeros(n_queries, self.args.TRAIN.WAY, device=queries.device)   # [35, 5]

        for label_idx, c in enumerate(unique_labels):
        
            # select keys and values for just this class
            class_k = torch.index_select(mh_support_set_ks, 0, extract_class_indices(support_labels, c))   # [1, 28, 1152]
            class_v = torch.index_select(mh_support_set_vs, 0, extract_class_indices(support_labels, c))   # [1, 28, 1152]
            k_bs = class_k.shape[0]

            class_scores = torch.matmul(mh_queries_ks.unsqueeze(1), class_k.transpose(-2,-1)) / math.sqrt(self.args.trans_linear_out_dim)   # ([35, 1, 28, 1152], [1, 1152, 28]) --> [35, 1, 28, 28]

            # reshape etc. to apply a softmax for each query tuple
            class_scores = class_scores.permute(0,2,1,3)     # [35, 28, 1, 28]
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1)     # [35, 28, 28]
            class_scores = [self.class_softmax(class_scores[i]) for i in range(n_queries)]
            class_scores = torch.cat(class_scores)      # [980, 28]
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1, self.tuples_len)    # [35, 28, 1, 28]
            class_scores = class_scores.permute(0,2,1,3)       # [35, 1, 28, 28]
            
            # get query specific class prototype         
            query_prototype = torch.matmul(class_scores, class_v)    # [35, 1, 28, 1152]
            query_prototype = torch.sum(query_prototype, dim=1)      # [35, 1, 28, 1152]
            
            # calculate distances from queries to query-specific class prototypes
            diff = mh_queries_vs - query_prototype   # [35, 28, 1152]
            norm_sq = torch.norm(diff, dim=[-2,-1])**2    # [35]
            distance = torch.div(norm_sq, self.tuples_len)
            
            # multiply by -1 to get logits
            distance = distance * -1
            c_idx = c.long()
            all_distances_tensor[:,c_idx] = distance
        
        return_dict = {'logits': all_distances_tensor}
        
        return return_dict


@HEAD_REGISTRY.register()
class CNN_TRX(CNN_FSHead):
    """
    Backbone connected to Temporal Cross Transformers of multiple cardinalities.
    """
    def __init__(self, cfg):
        super(CNN_TRX, self).__init__(cfg)
        args = cfg
        #fill default args
        self.args.trans_linear_out_dim = 1152
        self.args.temp_set = [2,3]
        self.args.trans_dropout = 0.1

        self.transformers = nn.ModuleList([TemporalCrossTransformer(args, s) for s in args.temp_set]) 

    def forward(self, inputs):
        # support_images, support_labels, target_images = inputs
        support_images, support_labels, target_images = inputs['support_set'], inputs['support_labels'], inputs['target_set'] # [200, 3, 224, 224]
        support_features, target_features = self.get_feats(support_images, target_images)
        all_logits = [t(support_features, support_labels, target_features)['logits'] for t in self.transformers]
        all_logits = torch.stack(all_logits, dim=-1)
        sample_logits = all_logits 
        sample_logits = torch.mean(sample_logits, dim=[-1])

        return_dict = {'logits': sample_logits}
        return return_dict

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs. Leaves TRX on GPU 0.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            self.backbone.cuda(0)
            self.backbone = torch.nn.DataParallel(self.backbone, device_ids=[i for i in range(0, self.args.num_gpus)])

            self.transformers.cuda(0)





def OTAM_cum_dist(dists, lbda=0.1):
    """
    Calculates the OTAM distances for sequences in one direction (e.g. query to support).
    :input: Tensor with frame similarity scores of shape [n_queries, n_support, query_seq_len, support_seq_len] 
    TODO: clearn up if possible - currently messy to work with pt1.8. Possibly due to stack operation?
    """
    dists = F.pad(dists, (1,1), 'constant', 0)  # [25, 25, 8, 10]

    cum_dists = torch.zeros(dists.shape, device=dists.device)

    # top row
    for m in range(1, dists.shape[3]):
        # cum_dists[:,:,0,m] = dists[:,:,0,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,0,m-1]))
        # paper does continuous relaxation of the cum_dists entry, but it trains faster without, so using the simpler version for now:
        cum_dists[:,:,0,m] = dists[:,:,0,m] + cum_dists[:,:,0,m-1] 


    # remaining rows
    for l in range(1,dists.shape[2]):
        #first non-zero column
        cum_dists[:,:,l,1] = dists[:,:,l,1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,0] / lbda) + torch.exp(- cum_dists[:,:,l-1,1] / lbda) + torch.exp(- cum_dists[:,:,l,0] / lbda) )
        
        #middle columns
        for m in range(2,dists.shape[3]-1):
            cum_dists[:,:,l,m] = dists[:,:,l,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,m-1] / lbda) + torch.exp(- cum_dists[:,:,l,m-1] / lbda ) )
            
        #last column
        #cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
        cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l-1,-1] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
    
    return cum_dists[:,:,-1,-1]


@HEAD_REGISTRY.register()
class CNN_OTAM(CNN_FSHead):
    """
    OTAM with a CNN backbone.
    """
    def __init__(self, cfg):
        super(CNN_OTAM, self).__init__(cfg)
        args = cfg
        self.args = cfg

    def forward(self, inputs):
        support_images, support_labels, target_images = inputs['support_set'], inputs['support_labels'], inputs['target_set'] # [200, 3, 224, 224]
        # [200, 3, 84, 84]

        support_features, target_features = self.get_feats(support_images, target_images)
        # [25, 8, 2048] [25, 8, 2048]
        unique_labels = torch.unique(support_labels)

        n_queries = target_features.shape[0]
        n_support = support_features.shape[0]

        support_features = rearrange(support_features, 'b s d -> (b s) d')  # [200, 2048]
        target_features = rearrange(target_features, 'b s d -> (b s) d')    # [200, 2048]

        frame_sim = cos_sim(target_features, support_features)    # [200, 200]
        frame_dists = 1 - frame_sim
        
        dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb = n_queries, sb = n_support)  # [25, 25, 8, 8]

        # calculate query -> support and support -> query
        cum_dists = OTAM_cum_dist(dists) + OTAM_cum_dist(rearrange(dists, 'tb sb ts ss -> tb sb ss ts'))


        class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(support_labels, c)), dim=1) for c in unique_labels]
        class_dists = torch.stack(class_dists)
        class_dists = rearrange(class_dists, 'c q -> q c')
        return_dict = {'logits': - class_dists}
        return return_dict

    def loss(self, task_dict, model_dict):
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())





@HEAD_REGISTRY.register()
class CNN_CrossTransformer(CNN_FSHead):
    """
    OTAM with a CNN backbone.
    """
    def __init__(self, cfg):
        super(CNN_CrossTransformer, self).__init__(cfg)
        args = cfg
        self.args = cfg
        self.dim = 2048
        # self.hidden_dim = 512  # v0 
        # self.hidden_dim = 2048   # v1
        self.hidden_dim = 1024   # v2
        self.way = cfg.TRAIN.WAY
        self.shot = cfg.TRAIN.SHOT
        self.key_head = nn.Conv1d(self.dim, self.hidden_dim, 1, bias=False)
        self.query_head = self.key_head
        self.value_head = nn.Conv1d(self.dim, self.hidden_dim, 1, bias=False)

    def forward(self, inputs):
        support_images, support_labels, target_images = inputs['support_set'], inputs['support_labels'], inputs['target_set'] # [200, 3, 224, 224]
        # [200, 3, 84, 84]
        # support_images, support_labels, target_images = inputs

        support_features, query_image_features = self.get_feats(support_images, target_images)
        # [25, 8, 2048] [25, 8, 2048]

        unique_labels = torch.unique(support_labels)
        support_features = [torch.index_select(support_features, 0, extract_class_indices(support_labels, c)) for c in unique_labels]
        support_features = torch.cat(support_features, 0)  # [25, 8, 2048]
     
        query = self.query_head(query_image_features.permute(0,2,1))   # [25, 512, 8]
        support_key = self.key_head(support_features.permute(0,2,1))
        support_value = self.value_head(support_features.permute(0,2,1))

        ## flatten pixels & k-shot in support (j & m in the paper respectively)
        support_key = support_key.view(self.way, self.shot, support_key.shape[1], -1)
        support_value = support_value.view(self.way, self.shot, support_value.shape[1], -1)

        support_key = support_key.permute(0, 2, 3, 1)
        support_value = support_value.permute(0, 2, 3, 1)

        support_key = support_key.contiguous().view(self.way, support_key.shape[1], -1)   # [5, 512, 40]
        support_value = support_value.contiguous().view(self.way, support_value.shape[1], -1)

		## v is j images' m pixels, ie k-shot*h*w
        attn_weights = torch.einsum('bdp,ndv->bnpv', query, support_key) * (self.hidden_dim ** -0.5)   # [15, 5, 8, 40]
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
		
        ## get weighted sum of support values
        support_value = support_value.unsqueeze(0).expand(attn_weights.shape[0], -1, -1, -1)   # [15, 5, 512, 40]
        query_aligned_prototype = torch.einsum('bnpv,bndv->bnpd', attn_weights, support_value)   # [15, 5, 8, 512]

		### Step 3: Calculate query value
        query_value = self.value_head(query_image_features.permute(0,2,1)).permute(0,2,1)   # [25, 8, 512]
		# query_value = query_value.view(query_value.shape[0], -1, query_value.shape[1]) ##bpd
		
		### Step 4: Calculate distance between queries and supports
        distances = []
        for classid in range(query_aligned_prototype.shape[1]):
            support_features = rearrange(F.normalize(query_aligned_prototype[:, classid], dim=2), 'b s d -> b (s d)')     # [15, 4096]
            target_features = rearrange(F.normalize(query_value, dim=2), 'b s d -> b (s d)')      # [15, 4096]
            # dxc = torch.matmul(target_features, support_features.transpose(0,1))
            dxc = (target_features*support_features).sum(1)/8   # vo is no/8

			# dxc = torch.cdist(query_aligned_prototype[:, classid], 
			# 								query_value, p=2)
			# dxc = dxc**2
			# B,P,R = dxc.shape
			# dxc = dxc.sum(dim=(1,2)) / (P*R)
            distances.append(dxc)
		
        # class_dists = rearrange(class_dists, 'c q -> q c')
        class_dists = torch.stack(distances, dim=1)
        return_dict = {'logits': class_dists}
        return return_dict

    def loss(self, task_dict, model_dict):
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())
    


@HEAD_REGISTRY.register()
class CNN_TSN(CNN_FSHead):
    """
    TSN with a CNN backbone.
    Either cosine similarity or negative norm squared distance. 
    Use mean distance from query to class videos.
    """
    def __init__(self, cfg):
        super(CNN_TSN, self).__init__(cfg)
        args = cfg
        self.norm_sq_dist = False


    def forward(self, inputs):
        # [200, 3, 224, 224] [4., 4., 0., 2., 0., 3., 3., 4., 3., 1., 4., 2., 2., 1., 1., 0., 2., 1., 1., 0., 3., 2., 0., 3., 4.]  [200, 3, 224, 224]
        support_images, support_labels, target_images = inputs['support_set'], inputs['support_labels'], inputs['target_set'] # [200, 3, 224, 224]
        support_features, target_features = self.get_feats(support_images, target_images)   # [25, 8, 2048] [25, 8, 2048]
        unique_labels = torch.unique(support_labels)

        support_features = torch.mean(support_features, dim=1)
        target_features = torch.mean(target_features, dim=1)

        if self.norm_sq_dist:
            class_prototypes = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
            class_prototypes = torch.stack(class_prototypes)
            
            diffs = [target_features - class_prototypes[i] for i in unique_labels]
            diffs = torch.stack(diffs)

            norm_sq = torch.norm(diffs, dim=[-1])**2
            distance = - rearrange(norm_sq, 'c q -> q c')
            return_dict = {'logits': distance}

        else:
            class_sim = cos_sim(target_features, support_features)
            class_sim = [torch.mean(torch.index_select(class_sim, 1, extract_class_indices(support_labels, c)), dim=1) for c in unique_labels]
            class_sim = torch.stack(class_sim)
            class_sim = rearrange(class_sim, 'c q -> q c')
            return_dict = {'logits': class_sim}

        return return_dict


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)



class PositionalEncoder(nn.Module):
    def __init__(self, d_model=2048, max_seq_len = 20, dropout = 0.1, A_scale=10., B_scale=1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.A_scale = A_scale
        self.B_scale = B_scale
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        
        x = x * math.sqrt(self.d_model/self.A_scale)
        #add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + self.B_scale * pe
        return self.dropout(x)


@HEAD_REGISTRY.register()
class CNN_HyRSM_1shot(CNN_FSHead):
    """
    OTAM with a CNN backbone.
    """

    def __init__(self, cfg):
        super(CNN_HyRSM_1shot, self).__init__(cfg)
        last_layer_idx = -1
        self.args = cfg
        
        self.relu = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)
        if self.args.VIDEO.HEAD.BACKBONE_NAME == "resnet50":
            self.mid_dim = 2048
        else:
            self.mid_dim = 512
        if hasattr(self.args.TRAIN,"POSITION_A") and hasattr(self.args.TRAIN,"POSITION_B"):
            self.pe = PositionalEncoder(d_model=self.mid_dim, dropout=0.1, A_scale=self.args.TRAIN.POSITION_A, B_scale=self.args.TRAIN.POSITION_B)
        else:
            self.pe = PositionalEncoder(d_model=self.mid_dim, dropout=0.1, A_scale=10., B_scale=1.)
        if hasattr(self.args.TRAIN,"HEAD") and self.args.TRAIN.HEAD:
            self.temporal_atte_before = PreNormattention(self.mid_dim, Attention(self.mid_dim, heads = self.args.TRAIN.HEAD, dim_head = self.mid_dim//self.args.TRAIN.HEAD, dropout = 0.2))
            self.temporal_atte = MultiHeadAttention(self.args.TRAIN.HEAD, self.mid_dim, self.mid_dim//self.args.TRAIN.HEAD, self.mid_dim//self.args.TRAIN.HEAD, dropout=0.05)
        else:
            self.temporal_atte_before = PreNormattention(self.mid_dim, Attention(self.mid_dim, heads = 8, dim_head = self.mid_dim//8, dropout = 0.2))
            self.temporal_atte = MultiHeadAttention(8, self.mid_dim, self.mid_dim//8, self.mid_dim//8, dropout=0.05)
        
        self.layer2 = nn.Sequential(nn.Conv1d(self.mid_dim*2, self.mid_dim, kernel_size=1, padding=0),)
                                   
        if hasattr(self.args.TRAIN, "NUM_CLASS"):
            self.classification_layer = nn.Linear(self.mid_dim, int(self.args.TRAIN.NUM_CLASS))
        else:
            self.classification_layer = nn.Linear(self.mid_dim, 64)

    def get_feats(self, support_images, target_images):
        """
        Takes in images from the support set and query video and returns CNN features.
        """
        support_features = self.backbone(support_images).squeeze()  # [40, 2048, 7, 7] (5 way - 1 shot - 5 query)
        target_features = self.backbone(target_images).squeeze()   # [200, 2048, 7, 7]
        # set_trace()
        batch_s = int(support_features.shape[0])
        batch_t = int(target_features.shape[0])

        dim = int(support_features.shape[1])

        Query_num = target_features.shape[0]//self.args.DATA.NUM_INPUT_FRAMES

        support_features = self.relu(self.temporal_atte_before(self.pe(support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim))))   # [35, 5, 8, 2048]  V1
        target_features = self.relu(self.temporal_atte_before(self.pe(target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim))))   # .repeat(1,self.args.TRAIN.WAY,1,1)  # [35, 1, 8, 2048]

        if hasattr(self.args.TRAIN, "NUM_CLASS"):
            class_logits = self.classification_layer(torch.cat([support_features, target_features], 0)).reshape(-1, int(self.args.TRAIN.NUM_CLASS))
        else:
            class_logits = self.classification_layer(torch.cat([support_features, target_features], 0)).reshape(-1, 64)
        support_features_ext = support_features.unsqueeze(0).repeat(Query_num,1,1,1)
        target_features_ext = target_features.unsqueeze(1)
        
        feature_in = torch.cat([support_features_ext.mean(2), target_features_ext.mean(2)], 1)
        
        feature_in = self.relu(self.temporal_atte(feature_in, feature_in, feature_in)) 
        support_features = torch.cat([support_features_ext, feature_in[:,:-1,:].unsqueeze(2).repeat(1,1,self.args.DATA.NUM_INPUT_FRAMES,1)], 3).reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim*2)
        support_features= self.layer2(support_features.permute(0,2,1)).permute(0,2,1).reshape(Query_num, -1, self.args.DATA.NUM_INPUT_FRAMES, dim)
        target_features = self.layer2(torch.cat([target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), feature_in[:,-1,:].unsqueeze(1).repeat(1,self.args.DATA.NUM_INPUT_FRAMES,1)],2).permute(0,2,1)).permute(0,2,1)

        return support_features, target_features, class_logits

    def forward(self, inputs):
        support_images, support_labels, target_images = inputs['support_set'], inputs['support_labels'], inputs['target_set'] # [200, 3, 224, 224]
        
        support_features, target_features, class_logits = self.get_feats(support_images, target_images)
        # [35, 5, 8, 2048] [35, 8, 2048] [40, 64]
        unique_labels = torch.unique(support_labels)

        n_queries = target_features.shape[0]
        n_support = support_features.shape[1]
        frame_num = support_features.shape[2]
        # F.normalize(support_features, dim=2)

        support_features = rearrange(support_features, 'b h s d -> b (h s) d')  # [200, 2048] [35, 40, 2048]
 
        frame_sim = torch.matmul(F.normalize(support_features, dim=2), F.normalize(target_features, dim=2).permute(0,2,1)).reshape(n_queries, n_support, frame_num, frame_num)
        frame_dists = 1 - frame_sim
        dists = frame_dists
        
        cum_dists = dists.min(3)[0].sum(2) + dists.min(2)[0].sum(2) 

        class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(support_labels, c)), dim=1) for c in unique_labels]
        class_dists = torch.stack(class_dists)   # [5, 35]
        class_dists = rearrange(class_dists, 'c q -> q c')
        return_dict = {'logits': - class_dists, 'class_logits': class_logits}
        return return_dict

    def loss(self, task_dict, model_dict):
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())





@HEAD_REGISTRY.register()
class CNN_HyRSM_5shot(CNN_FSHead):
    """
    TSN with a CNN backbone.
    Either cosine similarity or negative norm squared distance. 
    Use mean distance from query to class videos.
    """
    def __init__(self, cfg):
        super(CNN_HyRSM_5shot, self).__init__(cfg)
        args = cfg
        self.args = cfg
        self.norm_sq_dist = False
        if self.args.VIDEO.HEAD.BACKBONE_NAME == "resnet50":
            self.mid_dim = 2048
        else:
            self.mid_dim = 512
        if hasattr(self.args.TRAIN,"POSITION_A") and hasattr(self.args.TRAIN,"POSITION_B"):
            self.pe = PositionalEncoder(d_model=self.mid_dim, dropout=0.1, A_scale=self.args.TRAIN.POSITION_A, B_scale=self.args.TRAIN.POSITION_B)
        else:
            self.pe = PositionalEncoder(d_model=self.mid_dim, dropout=0.1, A_scale=10., B_scale=1.)
        
        last_layer_idx = -1
        
        self.relu = nn.ReLU(inplace=True)
        # self.relu1 = nn.ReLU(inplace=True)
        if hasattr(self.args.TRAIN,"HEAD") and self.args.TRAIN.HEAD:
            self.temporal_atte_before = PreNormattention(self.mid_dim, Attention(self.mid_dim, heads = self.args.TRAIN.HEAD, dim_head = self.mid_dim//self.args.TRAIN.HEAD, dropout = 0.2))
            self.temporal_atte = MultiHeadAttention(self.args.TRAIN.HEAD, self.mid_dim, self.mid_dim//self.args.TRAIN.HEAD, self.mid_dim//self.args.TRAIN.HEAD, dropout=0.05)
        else:
            self.temporal_atte_before = PreNormattention(self.mid_dim, Attention(self.mid_dim, heads = 8, dim_head = self.mid_dim//8, dropout = 0.2))
            self.temporal_atte = MultiHeadAttention(8, self.mid_dim, self.mid_dim//8, self.mid_dim//8, dropout=0.05)
        
        self.layer2 = nn.Sequential(nn.Conv1d(self.mid_dim*2, self.mid_dim, kernel_size=1, padding=0),)
                                    
        if hasattr(self.args.TRAIN, "NUM_CLASS"):
            self.classification_layer = nn.Linear(self.mid_dim, int(self.args.TRAIN.NUM_CLASS))
        else:
            self.classification_layer = nn.Linear(self.mid_dim, 64)

    def get_feats(self, support_images, target_images, support_labels):
        """
        Takes in images from the support set and query video and returns CNN features.
        """
        support_features = self.backbone(support_images).squeeze()  # [40, 2048, 7, 7] (5 way - 1 shot - 5 query)
        target_features = self.backbone(target_images).squeeze()   # [200, 2048, 7, 7]
        # set_trace()
        batch_s = int(support_features.shape[0])
        batch_t = int(target_features.shape[0])

        dim = int(support_features.shape[1])
        
        # Temporal
        Query_num = target_features.shape[0]//self.args.DATA.NUM_INPUT_FRAMES

        support_features = self.relu(self.temporal_atte_before(self.pe(support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim))))   # [25, 8, 2048]
        target_features = self.relu(self.temporal_atte_before(self.pe(target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim))))   # [15, 8, 2048]

        if hasattr(self.args.TRAIN, "NUM_CLASS"):
            class_logits = self.classification_layer(torch.cat([support_features, target_features], 0)).reshape(-1, int(self.args.TRAIN.NUM_CLASS))
        else:
            class_logits = self.classification_layer(torch.cat([support_features, target_features], 0)).reshape(-1, 64)

        unique_labels = torch.unique(support_labels)
        QUERY_PER_CLASS = target_features.shape[0]//self.args.TRAIN.WAY

        class_prototypes = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
        class_prototypes = torch.stack(class_prototypes)   # [5, 2048, 8]

        support_features_ext = class_prototypes.unsqueeze(0).repeat(Query_num,1,1,1)
        target_features_ext = target_features.unsqueeze(1)
        
        feature_in = torch.cat([support_features_ext.mean(2), target_features_ext.mean(2)], 1)
        # feature_in = self.temporal_atte(feature_in, feature_in, feature_in)  # .view(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)) [35, 6, 2048]  45%
        feature_in = self.relu(self.temporal_atte(feature_in, feature_in, feature_in)) 
        support_features = torch.cat([support_features_ext, feature_in[:,:-1,:].unsqueeze(2).repeat(1,1,self.args.DATA.NUM_INPUT_FRAMES,1)], 3).reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim*2)
        support_features= self.layer2(support_features.permute(0,2,1)).permute(0,2,1).reshape(Query_num, -1, self.args.DATA.NUM_INPUT_FRAMES, dim)
        target_features = self.layer2(torch.cat([target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), feature_in[:,-1,:].unsqueeze(1).repeat(1,self.args.DATA.NUM_INPUT_FRAMES,1)],2).permute(0,2,1)).permute(0,2,1)

        return support_features, target_features, class_logits


    def forward(self, inputs):
        # [200, 3, 224, 224] [4., 4., 0., 2., 0., 3., 3., 4., 3., 1., 4., 2., 2., 1., 1., 0., 2., 1., 1., 0., 3., 2., 0., 3., 4.]  [200, 3, 224, 224]
        support_images, support_labels, target_images = inputs['support_set'], inputs['support_labels'], inputs['target_set'] # [200, 3, 224, 224]
        
        support_features, target_features, class_logits = self.get_feats(support_images, target_images, support_labels)
        # [35, 5, 8, 2048] [35, 8, 2048] [40, 64]
        unique_labels = torch.unique(support_labels)

        n_queries = target_features.shape[0]
        n_support = support_features.shape[1]
        frame_num = support_features.shape[2]
        # F.normalize(support_features, dim=2)

        support_features = rearrange(support_features, 'b h s d -> b (h s) d')  # [200, 2048] [35, 40, 2048]
        # target_features = rearrange(target_features, 'b s d -> (b s) d')    # [200, 2048]   [280, 2048]
 
        # frame_sim = cos_sim(target_features, support_features)    # [200, 200]
        frame_sim = torch.matmul(F.normalize(support_features, dim=2), F.normalize(target_features, dim=2).permute(0,2,1)).reshape(n_queries, n_support, frame_num, frame_num)
        frame_dists = 1 - frame_sim
        dists = frame_dists

        # calculate query -> support and support -> query
        cum_dists = dists.min(3)[0].sum(2) + dists.min(2)[0].sum(2) 
        
        class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
        class_dists = torch.stack(class_dists)   # [5, 35]
        class_dists = rearrange(class_dists, 'c q -> q c')
        return_dict = {'logits': - class_dists, 'class_logits': class_logits}
        return return_dict




