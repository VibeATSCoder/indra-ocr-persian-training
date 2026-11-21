"""Compute CTC and distillation per GPU; communicate four sums, not logits."""
import torch
from torch.nn import functional as F

class LocalObjective(torch.nn.Module):
    def __init__(self,student,teacher,classes=1356):
        super().__init__();self.student=student;self.teacher=teacher;self.classes=classes
    def train(self,mode=True):
        super().train(mode);self.teacher.eval();return self
    def forward(self,x,widths,targets,sizes):
        logits,lengths=self.student(x,widths)
        with torch.no_grad():old,_=self.teacher(x,widths)
        losses=F.ctc_loss(logits.float().log_softmax(-1).transpose(0,1),targets,lengths.cpu(),sizes.cpu(),reduction='none',zero_infinity=False)
        mask=torch.arange(logits.shape[1],device=x.device)[None,:]<lengths[:,None]
        kl=(F.kl_div(F.log_softmax(logits[:,:,:self.classes].float()/2,-1),F.softmax(old[:,:,:self.classes].float()/2,-1),reduction='none').sum(-1)*mask).sum()*4
        return torch.stack([(losses/sizes).sum(),losses.new_tensor(len(x)),kl,mask.sum()])[None,:]
