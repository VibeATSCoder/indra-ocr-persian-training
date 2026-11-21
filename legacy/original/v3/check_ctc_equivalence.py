"""Compare original and blank-padded CTC objectives and valid-frame gradients."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from ctc_objective import padded_ctc
torch.set_num_threads(2);torch.manual_seed(907)
checks=[]
for classes,frames,lengths,sizes in [
    (48,128,[128,103,91,71],[41,31,29,17]),
    (5,12,[3,8,12],[2,3,4]),
    (1357,64,[64,49,38],[20,17,12])]:
    sizes=torch.tensor(sizes);lengths=torch.tensor(lengths)
    labels=torch.cat([torch.randint(1,classes,(int(n),)) for n in sizes])
    if classes==5:labels[:2]=1
    x=torch.randn(len(lengths),frames,classes,requires_grad=True)
    original=F.ctc_loss(x.log_softmax(-1).transpose(0,1),labels,lengths,sizes)
    original.backward();gradient=x.grad.detach().clone();x.grad=None
    padded=padded_ctc(x,lengths,labels,sizes);padded.backward()
    torch.testing.assert_close(padded,original,atol=2e-5,rtol=1e-5)
    torch.testing.assert_close(x.grad,gradient,atol=2e-5,rtol=1e-3)
    checks.append({'classes':classes,'frames':frames,'loss_delta':abs(padded.item()-original.item()),'max_gradient_delta':float((x.grad-gradient).abs().max())})
report={'checks_passed':True,'cases':checks}
(Path(__file__).parent/'ctc-equivalence-cpu.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
