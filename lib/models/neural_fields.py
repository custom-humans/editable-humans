import torch
import torch.nn.functional as F
import torch.nn as nn
import logging as log

from .feature_dictionary import FeatureDictionary
from .networks.positional_encoding import PositionalEncoding
from .networks.mlps import MLP, Conditional_MLP
from .networks.layers import get_layer_class


def get_activation_class(activation_type):
    """Utility function to return an activation function class based on the string description.

    Args:
        activation_type (str): The name for the activation function.
    
    Returns:
        (Function): The activation function to be used. 
    """
    if activation_type == 'relu':
        return torch.relu
    elif activation_type == 'sin':
        return torch.sin
    elif activation_type == 'softplus':
        return torch.nn.functional.softplus
    elif activation_type == 'lrelu':
        return torch.nn.functional.leaky_relu
    else:
        assert False and "activation type does not exist"


####################################################
class NeuralField(nn.Module):

    def __init__(self,
        cfg          :dict,
        smpl_V       :torch.Tensor,
        smpl_F       :torch.Tensor,
        feat_dim     : int,
        out_dim      : int,
        pos_freq     : int,
        low_rank     : int,
        sigmoid      : bool = False,
    ):
        
        super().__init__()
        self.cfg = cfg
        self.smpl_V = smpl_V
        self.smpl_F = smpl_F
        self.feat_dim = feat_dim
        self.out_dim = out_dim
        self.pos_freq = pos_freq
        self.low_rank = low_rank
        self.sigmoid = sigmoid

        self.pos_dim = self.cfg.pos_dim
        self.c_dim = self.cfg.c_dim
        self.activation = self.cfg.activation
        self.layer_type = self.cfg.layer_type
        self.hidden_dim = self.cfg.hidden_dim
        self.num_layers = self.cfg.num_layers
        self.skip = self.cfg.skip
        self.feature_std = self.cfg.feature_std
        self.feature_bias = self.cfg.feature_bias


        self._init_dictionary()
        self._init_embedder()
        self._init_decoder()


    def _init_dictionary(self):
        """Initialize the feature dictionary object.
        """

        self.dictionary = FeatureDictionary(self.feat_dim, self.feature_std, self.feature_bias)
        self.dictionary.init_from_smpl_vertices(self.smpl_V)

    def _init_embedder(self):
        """Initialize positional embedding objects.
        """
        self.embedder = PositionalEncoding(self.pos_freq, self.pos_freq -1, input_dim=self.pos_dim)
        self.embed_dim = self.embedder.out_dim

    def _init_decoder(self):
        """Initialize the decoder object.
        """
        self.input_dim = self.embed_dim + self.feat_dim

        if self.c_dim <= 0:
            self.decoder = MLP(self.input_dim, self.out_dim, activation=get_activation_class(self.activation),
                                    bias=True, layer=get_layer_class(self.layer_type), num_layers=self.num_layers,
                                    hidden_dim=self.hidden_dim, skip=self.skip)
        else:
            self.decoder = Conditional_MLP(self.input_dim, self.c_dim, self.out_dim,  activation=get_activation_class(self.activation),
                                        bias=True, layer=get_layer_class(self.layer_type), num_layers=self.num_layers,
                                        hidden_dim=self.hidden_dim, skip=self.skip)    


        log.info("Total number of parameters {}".format(
            sum(p.numel() for p in self.decoder.parameters()))\
        )

    def forward_decoder(self, feats, local_coords, normal, return_h=False, f=None):
        """Forward pass through the MLP decoder.
            Args:
                feats (torch.FloatTensor): Feature tensor of shape [B, N, feat_dim]
                local_coords (torch.FloatTensor): Local coordinate tensor of shape [B, N, 3]
                normal (torch.FloatTensor): Normal tensor of shape [B, N, 3]
                return_h (bool): Whether to return the hidden states of the network.
                f (torch.FloatTensor): The conditional feature tensor of shape [B, c_dim]
        
        """

        if self.c_dim <= 0:
            input = torch.cat([self.embedder(local_coords), feats], dim=-1)
            return self.decoder(input, return_h=return_h, sigmoid=self.sigmoid)
        else:
            input = torch.cat([self.embedder(local_coords), feats], dim=-1)
            if f is not None:
                c = torch.cat([f, normal], dim=-1)
            else:
                c = normal
            return self.decoder(input, c, return_h=return_h, sigmoid=self.sigmoid)
        

    def forward(self, x, code_idx, pose_idx=None, return_h=False, f=None):
        """Forward pass through the network.
            Args:
                x (torch.FloatTensor): Coordinate tensor of shape [B, N, 3]
                code_idx (torch.LongTensor): Code index tensor of shape [B, 1]
                pose_idx (torch.LongTensor): SMPL_V index tensor of shape [B, 1]
                return_h (bool): Whether to return the hidden states of the network.
                f (torch.FloatTensor): The conditional feature tensor of shape [B, c_dim]
        """
        if pose_idx is None:
            pose_idx = code_idx
        feats, local_coords, normal = self.dictionary.interpolate(x, code_idx, self.smpl_V[pose_idx], self.smpl_F)

        return self.forward_decoder(feats, local_coords, normal, return_h=return_h, f=f)
        
    def sample(self, x, idx, return_h=False, f=None):
        """Sample from the network.
        """
        feats, local_coords, normal = self.dictionary.interpolate_random(x, self.smpl_V[idx], self.smpl_F, self.low_rank)

        return self.forward_decoder(feats, local_coords, normal, return_h=return_h, f=f)

            
    def regularization_loss(self, idx=None):
        """Compute the L2 regularization loss.
        """

        if idx is None:
            return (self.dictionary.feature_codebooks**2).mean()
        else:
            return (self.dictionary.feature_codebooks[idx]**2).mean()


    def finitediff_gradient(self, x, idx, eps=0.005, sample=False):
        """Compute 3D gradient using finite difference.

        Args:
            x (torch.FloatTensor): Coordinate tensor of shape [B, N, 3]
        """
        shape = x.shape

        eps_x = torch.tensor([eps, 0.0, 0.0], device=x.device)
        eps_y = torch.tensor([0.0, eps, 0.0], device=x.device)
        eps_z = torch.tensor([0.0, 0.0, eps], device=x.device)

        # shape: [B, 6, N, 3] -> [B, 6*N, 3]
        x_new = torch.stack([x + eps_x, x + eps_y, x + eps_z,
                           x - eps_x, x - eps_y, x - eps_z], dim=1).reshape(shape[0], -1, shape[-1])
        
        # shape: [B, 6*N, 3] -> [B, 6, N, 3]
        if sample:
            pred = self.sample(x_new, idx).reshape(shape[0], 6, -1)
        else:
            pred = self.forward(x_new, idx).reshape(shape[0], 6, -1)
        grad_x = (pred[:, 0, ...] - pred[:, 3, ...]) / (eps * 2.0)
        grad_y = (pred[:, 1, ...] - pred[:, 4, ...]) / (eps * 2.0)
        grad_z = (pred[:, 2, ...] - pred[:, 5, ...]) / (eps * 2.0)

        return torch.stack([grad_x, grad_y, grad_z], dim=-1)
    
    
    def forward_fitting(self, x, code, smpl_V, return_h=False, f=None):
        """Forward pass through the network with a latent code input.
            Args:
                x (torch.FloatTensor): Coordinate tensor of shape [1, N, 3]
                code (torch.FloatTensor): Latent code tensor of shape [1, n_vertices, c_dim]
                smpl_V (torch.FloatTensor): SMPL_V tensor of shape [1, n_vertices, 3]
        """

        feats, local_coords, normal = self.dictionary.interpolate(x, 0, smpl_V, self.smpl_F, input_code=code)

        return self.forward_decoder(feats, local_coords, normal, return_h=return_h, f=f)

    def normal_fitting(self, x, code, smpl_V, eps=0.005):
        shape = x.shape

        eps_x = torch.tensor([eps, 0.0, 0.0], device=x.device)
        eps_y = torch.tensor([0.0, eps, 0.0], device=x.device)
        eps_z = torch.tensor([0.0, 0.0, eps], device=x.device)

        # shape: [B, 6, N, 3] -> [B, 6*N, 3]
        x_new = torch.stack([x + eps_x, x + eps_y, x + eps_z,
                           x - eps_x, x - eps_y, x - eps_z], dim=1).reshape(shape[0], -1, shape[-1])
        
        pred = self.forward_fitting(x_new, code, smpl_V).reshape(shape[0], 6, -1)
        grad_x = (pred[:, 0, ...] - pred[:, 3, ...]) / (eps * 2.0)
        grad_y = (pred[:, 1, ...] - pred[:, 4, ...]) / (eps * 2.0)
        grad_z = (pred[:, 2, ...] - pred[:, 5, ...]) / (eps * 2.0)

        return torch.stack([grad_x, grad_y, grad_z], dim=-1)

    def get_mean_feature(self, vert_idx=None):
        if vert_idx is None:
            return self.dictionary.feature_codebooks.mean(dim=0)
        else:
            return self.dictionary.feature_codebooks[:, vert_idx].mean(dim=0)

    def get_feature_by_idx(self, idx, vert_idx=None):
        if vert_idx is None:
            return self.dictionary.feature_codebooks[idx]
        else:
            return self.dictionary.feature_codebooks[idx][vert_idx]
    
    def replace_feature_by_idx(self, idx, feature, vert_idx=None):
        if vert_idx is None:
            self.dictionary.feature_codebooks[idx] = feature
        else:
            self.dictionary.feature_codebooks[idx][vert_idx] = feature

    def get_smpl_vertices_by_idx(self, idx):
        return self.smpl_V[idx]

    def replace_smpl_vertices_by_idx(self, idx, smpl_V):
        self.smpl_V[idx] = smpl_V
