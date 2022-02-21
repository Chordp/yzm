import torch
import torch.nn as nn
from torch.utils.data import dataloader
import CaptchaSet
import Net
import Utils
import string
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from torchvision.transforms.functional import to_pil_image
batch_size = 32
captcha_lable = '_' + string.ascii_lowercase + string.digits
n_classes = len(captcha_lable)
# width, height, n_input_length, n_len = 160, 60, 10, 4
# train_set = CaptchaSet.CaptchaDataset(captcha_lable, 1000 * batch_size, width, height, n_input_length, n_len)
# test_set = CaptchaSet.CaptchaDataset(captcha_lable, 100 * batch_size, width, height, n_input_length, n_len)
train_set = CaptchaSet.CaptchaSet(root_dir='./data/train')
test_set = CaptchaSet.CaptchaSet(root_dir='./data/test')
# ppp = train_set[0]
# ppp = to_pil_image(ppp[0])
# ppp.show()

trainDataload = dataloader.DataLoader(train_set,batch_size=64,shuffle=True,num_workers=0)
testDataload = dataloader.DataLoader(test_set,batch_size=64,shuffle=False,num_workers = 0)

loss_fun = nn.CTCLoss(blank=0, reduction='mean').cuda()
# loss_fun = nn.CTCLoss().cuda()
# net = Net.CRNN_OCR(n_classes=n_classes).cuda()
net = torch.load('bbb.pth').cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001,
                               betas=(0.5, 0.999))
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                            milestones=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                                            gamma=0.65)

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


def train(model, optimizer, epoch, dataloader,characters):
    model.train()
    loss_mean = 0
    acc_mean = 0
    with tqdm(dataloader) as pbar:
        for batch_index, (data, target, input_lengths, target_lengths) in enumerate(pbar):
            data, target = data.cuda(), target.cuda()
            
            optimizer.zero_grad()
            output = model(data)
            
            output_log_softmax = F.log_softmax(output, dim=-1)
            loss = loss_fun(output_log_softmax, target, input_lengths, target_lengths)
            
            loss.backward()
            optimizer.step()

            loss = loss.item()
            acc = calc_acc(target, output,characters)
            
            if batch_index == 0:
                loss_mean = loss
                acc_mean = acc
            
            loss_mean = 0.1 * loss + 0.9 * loss_mean
            acc_mean = 0.1 * acc + 0.9 * acc_mean
            
            pbar.set_description(f'Epoch: {epoch} Loss: {loss_mean:.4f} Acc: {acc_mean:.4f} ')

def valid(model, optimizer, epoch, dataloader,characters):
    model.eval()
    with tqdm(dataloader) as pbar, torch.no_grad():
        loss_sum = 0
        acc_sum = 0
        for batch_index, (data, target, input_lengths, target_lengths) in enumerate(pbar):
            data, target = data.cuda(), target.cuda()
            
            output = model(data)
            output_log_softmax = F.log_softmax(output, dim=-1)
            loss = loss_fun(output_log_softmax, target, input_lengths, target_lengths)
            
            loss = loss.item()
            acc = calc_acc(target, output,characters)
            
            loss_sum += loss
            acc_sum += acc
            
            loss_mean = loss_sum / (batch_index + 1)
            acc_mean = acc_sum / (batch_index + 1)
            
            pbar.set_description(f'Test : {epoch} Loss: {loss_mean:.4f} Acc: {acc_mean:.4f} ')
            


for epoch in range(1,101):
    train(net, optimizer, epoch, trainDataload,captcha_lable)
    valid(net, optimizer, epoch, testDataload,captcha_lable)
    if epoch % 3 == 0:
        scheduler.step()
        torch.save(net, 'bbb.pth')

