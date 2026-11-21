"""Equivalent blank padding makes variable-length CTC eligible for cuDNN."""
import torch
from torch.nn import functional as F

def padded_ctc(logits,lengths,labels,sizes):
    log_probs=logits.float().log_softmax(-1)
    valid=torch.arange(logits.shape[1],device=logits.device)[None,:]<lengths[:,None]
    blank=torch.full_like(log_probs,-10000);blank[:,:,0]=0
    log_probs=torch.where(valid[:,:,None],log_probs,blank).transpose(0,1).contiguous()
    full=torch.full((len(lengths),),logits.shape[1],dtype=torch.int32)
    return F.ctc_loss(log_probs,labels.to(device=logits.device,dtype=torch.int32),full,sizes.to(device='cpu',dtype=torch.int32),blank=0,zero_infinity=False)
