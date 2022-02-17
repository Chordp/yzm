import torch
import torch.nn as nn
from torch.utils.data import dataloader
import CaptchaSet
import Net
import Utils
import string
characters = '_' + string.ascii_lowercase + string.digits
def decode(sequence,characters):
    a = ''.join([characters[x] for x in sequence])
    s = ''.join([x for j, x in enumerate(a[:-1]) if x != characters[0] and x != a[j+1]])
    if len(s) == 0:
        return ''
    if a[-1] != characters[0] and s[-1] != a[-1]:
        s += a[-1]
    return s

def decode_target(sequence,characters):
    return ''.join([characters[x] for x in sequence]).replace(' ', '')

def calc_acc(target, output,characters):
    output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
    target = target.cpu().numpy()
    output_argmax = output_argmax.cpu().numpy()
    a = np.array([decode_target(true,characters) == decode(pred,characters) for true, pred in zip(target, output_argmax)])
    return a.mean()


test_set = CaptchaSet.CaptchaSet(root_dir='./data/train')
model = torch.load('bbb.pth').cpu()
model.eval()

# torch.onnx.export(model, torch.randn(1, 3, 60, 160), 'captcha.onnx', verbose=True)

for image,target,_,_ in test_set:
    output = model(image.unsqueeze(0))

    output_argmax = output.detach().permute(1,0,2).argmax(dim=-1)
    a,b = decode(target,characters),decode(output_argmax[0],characters)
    print(f'true:{a} pred:{b} Is{a==b}')



