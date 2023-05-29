""" The code is based on https://github.com/apple/ml-gsn/ with adaption. """

import torch
import torch.nn as nn
from torch import autograd
import logging as log
import torch.nn.functional as F

from .networks.discriminator import StyleDiscriminator

def hinge_loss(fake_pred, real_pred, mode):
    if mode == 'd':
        # Discriminator update
        d_loss_fake = F.relu(1.0 + fake_pred).mean()
        d_loss_real = F.relu(1.0 - real_pred).mean()
        d_loss = d_loss_fake + d_loss_real
    elif mode == 'g':
        # Generator update
        d_loss = -torch.mean(fake_pred)
    return d_loss

def logistic_loss(fake_pred, real_pred, mode):
    if mode == 'd':
        # Discriminator update
        d_loss_fake = F.softplus(fake_pred).mean()
        d_loss_real = F.softplus(-real_pred).mean()
        d_loss = d_loss_fake + d_loss_real
    elif mode == 'g':
        # Generator update
        d_loss = F.softplus(-fake_pred).mean()
    return d_loss


def r1_loss(real_pred, real_img):
    (grad_real,) = autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


class GANLoss(nn.Module):
    def __init__(
        self,
        cfg,
        disc_loss='logistic',
        auxillary=False
    ):
        super().__init__()


        self.cfg = cfg
        self.discriminator = StyleDiscriminator(3, 128, auxilary=auxillary)
        log.info("Total number of parameters {}".format(
            sum(p.numel() for p in self.discriminator.parameters()))\
        )

        if disc_loss == 'hinge':
            self.disc_loss = hinge_loss
        elif disc_loss == 'logistic':
            self.disc_loss = logistic_loss

        self.auxillary = auxillary

    def forward(self, disc_in_real, disc_in_fake, mode='g', gt_label=None):

        if mode == 'g':  # optimize generator
            loss = 0
            log = {}
            if self.auxillary:
                logits_fake, _ = self.discriminator(disc_in_fake)
            else:
                logits_fake = self.discriminator(disc_in_fake)

            g_loss = self.disc_loss(logits_fake, None, mode='g')
            log["loss_train/g_loss"] = g_loss.item()
            loss += g_loss * self.cfg.lambda_gan

            return loss, log

        if mode == 'd' :  # optimize discriminator
            if self.auxillary:
                logits_real, aux_real = self.discriminator(disc_in_real)
                logits_fake, aux_fake = self.discriminator(disc_in_fake.detach().clone())
            else:
                logits_real = self.discriminator(disc_in_real)
                logits_fake = self.discriminator(disc_in_fake.detach().clone())

            disc_loss = self.disc_loss(fake_pred=logits_fake, real_pred=logits_real, mode='d')

            # lazy regularization so we don't need to compute grad penalty every iteration
            if self.cfg.lambda_grad > 0:
                grad_penalty = r1_loss(logits_real, disc_in_real)

                # the 0 * logits_real is to trigger DDP allgather
                # https://github.com/rosinality/stylegan2-pytorch/issues/76
                grad_penalty = grad_penalty + (0 * logits_real.sum())
            else:
                grad_penalty = torch.tensor(0.0).type_as(disc_loss)

            d_loss = disc_loss * self.cfg.lambda_gan + grad_penalty * self.cfg.lambda_grad / 2
            if self.auxillary:
                d_loss += F.cross_entropy(aux_real, gt_label)
                d_loss += F.cross_entropy(aux_fake, gt_label)

            log = {
                "loss_train/disc_loss": disc_loss.item(),
                "loss_train/r1_loss": grad_penalty.item(),
                "loss_train/logits_real": logits_real.mean().item(),
                "loss_train/logits_fake": logits_fake.mean().item(),
            }

            return d_loss, log
