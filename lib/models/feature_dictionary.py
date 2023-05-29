import torch
import torch.nn.functional as F
import torch.nn as nn
import logging as log
from ..ops.mesh import *


class FeatureDictionary(nn.Module):

    def __init__(self, 
        feature_dim        : int,
        feature_std        : float = 0.1,
        feature_bias       : float = 0.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.feature_std = feature_std
        self.feature_bias = feature_bias

    def init_from_smpl_vertices(self, smpl_vertices):
      
        self.num_subjets = smpl_vertices.shape[0]
        self.num_vertices = smpl_vertices.shape[1]

        # Initialize feature codebooks
        fts = torch.zeros(self.num_subjets,self.num_vertices, self.feature_dim) + self.feature_bias
        fts += torch.randn_like(fts) * self.feature_std
        self.feature_codebooks = nn.Parameter(fts)

        log.info(f"Initalized feature codebooks with shape {self.feature_codebooks.shape}")

    def interpolate(self, coords, idx, smpl_V, smpl_F, input_code=None):

        """Query local features using the feature codebook, or the given input_code.
        Args:
            coords (torch.FloatTensor): coords of shape [batch, num_samples, 3]
            idx (torch.LongTensor): index of shape [batch, 1]
            smpl_V (torch.FloatTensor): SMPL vertices of shape [batch, num_vertices, 3]
            smpl_F (torch.LongTensor): SMPL faces of shape [num_faces, 3]
            input_code (torch.FloatTensor): input code of shape [batch, num_vertices, feature_dim]
        Returns:
            (torch.FloatTensor): interpolated features of shape [batch, num_samples, feature_dim]
        """

        sdf, hitpt, fid, weights = batched_closest_point_fast(smpl_V, smpl_F,
                                                              coords) # [B, Ns, 1], [B, Ns, 3], [B, Ns, 1], [B, Ns, 3]
        
        normal = torch.nn.functional.normalize( hitpt - coords, eps=1e-6, dim=2) # [B x Ns x 3]
        hitface = smpl_F[fid] # [B, Ns, 3]

        if input_code is None:
            inputs_feat = self.feature_codebooks[idx].unsqueeze(2).expand(-1, -1, hitface.shape[-1], -1) 
        else:
            inputs_feat = input_code.unsqueeze(2).expand(-1, -1, hitface.shape[-1], -1)
            
        indices = hitface.unsqueeze(-1).expand(-1, -1, -1, inputs_feat.shape[-1])
        nearest_feats = torch.gather(input=inputs_feat, index=indices, dim=1) # [B, Ns, 3, D]

        weighted_feats = torch.sum(nearest_feats * weights[...,None], dim=2) # K-weighted sum by: [B x Ns x 32]
        
        coords_feats = torch.cat([weights[...,1:], sdf], dim=-1) # [B, Ns, 3]
        return weighted_feats, coords_feats, normal
    
    def interpolate_random(self, coords, smpl_V, smpl_F, low_rank=32):
        """Query local features using PCA random sampling.

        Args:
            coords (torch.FloatTensor): coords of shape [batch, num_samples, 3]
            smpl_V (torch.FloatTensor): SMPL vertices of shape [batch, num_vertices, 3]
            smpl_F (torch.LongTensor): SMPL faces of shape [num_faces, 3]

        Returns:
            (torch.FloatTensor): interpolated features of shape [batch, num_samples, feature_dim]
        """
        b_size = coords.shape[0]

        sdf, hitpt, fid, weights = batched_closest_point_fast(smpl_V, smpl_F,
                                                              coords) # [B, Ns, 1], [B, Ns, 3], [B, Ns, 1], [B, Ns, 3]
        normal = torch.nn.functional.normalize( hitpt - coords, eps=1e-6, dim=2) # [B x Ns x 3]
        hitface = smpl_F[fid] # [B, Ns, 3]
        inputs_feat = self._pca_sample(low_rank=low_rank, batch_size=b_size).unsqueeze(2).expand(-1, -1, hitface.shape[-1], -1) 
        indices = hitface.unsqueeze(-1).expand(-1, -1, -1, inputs_feat.shape[-1])
        nearest_feats = torch.gather(input=inputs_feat, index=indices, dim=1) # [B, Ns, 3, D]

        weighted_feats = torch.sum(nearest_feats * weights[...,None], dim=2) # K-weighted sum by: [B x Ns x 32]
        
        coords_feats = torch.cat([weights[...,1:], sdf], dim=-1) # [B, Ns, 3]
        return weighted_feats, coords_feats, normal


    def _pca_sample(self, low_rank=32, batch_size=1):

        A = self.feature_codebooks.clone()
        num_subjects, num_vertices, dim = A.shape

        A = A.view(num_subjects, -1)

        (U, S, V) = torch.pca_lowrank(A, q=low_rank, center=True, niter=1)

        params = torch.matmul(A, V) # (N, 128)
        mean = params.mean(dim=0)
        cov = torch.cov(params.T)

        m = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov)
        random_codes = m.sample((batch_size,)).to(self.feature_codebooks.device)

        return torch.matmul(random_codes.detach(), V.t()).view(-1, num_vertices, dim)

