import torch
import torch.nn.functional as F
import torch.nn as nn


class SDFTracer(object):

    def __init__(self,
        cfg                  = None,
        camera_clamp : list  = [-4, 4],
        step_size    : float = 1.0,
        num_steps    : int   = 64, # samples for raymaching, iterations for sphere trace
        min_dis      : float = 1e-3): 

        self.camera_clamp = camera_clamp
        self.step_size = step_size
        self.num_steps = num_steps
        self.min_dis = min_dis

        self.inv_num_steps = 1.0 / self.num_steps

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, nef, idx, ray_o, ray_d):
        """PyTorch implementation of sphere tracing.
            Args:
                nef: Neural field object
                idx: index of the subject, shape (B, )
                ray_o: ray origin, shape (B, N, 3)
                ray_d: ray direction, shape (B, N, 3)
        """

        # Distanace from ray origin
        t = torch.zeros(ray_o.shape[0], ray_o.shape[1], 1, device=ray_o.device)

        # Position in model space
        x = torch.addcmul(ray_o, ray_d, t)

        cond = torch.ones_like(t).bool()
        
        normal = torch.zeros_like(x)
        # This function is in fact differentiable, but we treat it as if it's not, because
        # it evaluates a very long chain of recursive neural networks (essentially a NN with depth of
        # ~1600 layers or so). This is not sustainable in terms of memory use, so we return the final hit
        # locations, where additional quantities (normal, depth, segmentation) can be determined. The
        # gradients will propagate only to these locations. 
        with torch.no_grad():

            d = nef(x, idx)
            
            dprev = d.clone()

            # If cond is TRUE, then the corresponding ray has not hit yet.
            # OR, the corresponding ray has exit the clipping plane.
            #cond = torch.ones_like(d).bool()[:,0]

            # If miss is TRUE, then the corresponding ray has missed entirely.
            hit = torch.zeros_like(d).bool()
            
            for i in range(self.num_steps):
                # 1. Check if ray hits.
                #hit = (torch.abs(d) < self._MIN_DIS)[:,0] 
                # 2. Check that the sphere tracing is not oscillating
                #hit = hit | (torch.abs((d + dprev) / 2.0) < self._MIN_DIS * 3)[:,0]
                
                # 3. Check that the ray has not exit the far clipping plane.
                #cond = (torch.abs(t) < self.clamp[1])[:,0]
                
                hit = (torch.abs(t) < self.camera_clamp[1])
                
                # 1. not hit surface
                cond = cond & (torch.abs(d) > self.min_dis)

                # 2. not oscillating
                cond = cond & (torch.abs((d + dprev) / 2.0) > self.min_dis * 3)
                
                # 3. not a hit
                cond = cond & hit
                
                #cond = cond & ~hit
                
                # If the sum is 0, that means that all rays have hit, or missed.
                if not cond.any():
                    break

                # Advance the x, by updating with a new t
                x = torch.where(cond, torch.addcmul(ray_o, ray_d, t), x)
                
                # Store the previous distance
                dprev = torch.where(cond, d, dprev)

                # Update the distance to surface at x
                d[cond] = nef(x, idx)[cond] * self.step_size

                # Update the distance from origin 
                t = torch.where(cond, t+d, t)
    
        # AABB cull 

        hit = hit & ~(torch.abs(x) > 1.0).any(dim=-1,keepdim=True)
        #hit = torch.ones_like(d).byte()[...,0]
        
        # The function will return 
        #  x: the final model-space coordinate of the render
        #  t: the final distance from origin
        #  d: the final distance value from
        #  miss: a vector containing bools of whether each ray was a hit or miss
        
        #if hit.any():
        #    grad = nef.finitediff_gradient(x[hit], idx)
        #    _normal = F.normalize(grad, p=2, dim=-1, eps=1e-5)
        #    normal[hit] = _normal
        
        return x, hit
