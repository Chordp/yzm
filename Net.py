import torch.nn as nn
import torch
import torchvision.models as models

class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)  # *2因为使用双向LSTM，两个方向隐层单元拼在一起

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class CRNN_OCR(nn.Module):
    def __init__(self,hidden_size = 512,n_classes = 37):
        super(CRNN_OCR, self).__init__()
        
        self.cnn = models.mobilenet_v3_small(pretrained =True).features
        # self.rnn = nn.Sequential(
        #     BidirectionalLSTM(576, hidden_size, hidden_size),
        #     BidirectionalLSTM(hidden_size, hidden_size, n_classes)
        # )

        self.rnn = nn.LSTM(input_size=576, hidden_size=hidden_size, num_layers=2, bidirectional=True,dropout=0.2)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, None))
        # self.fc = nn.Sequential(
        #     nn.Linear(hidden_size*2 , 512),
        #     nn.Hardswish(),
        #     # nn.Dropout(p=0.2, inplace=True),
        #     nn.Linear(512 , n_classes)
        # )
        self.fc = nn.Linear(hidden_size*2 , n_classes)
    def forward(self, x):
        conv = self.cnn(x)
        # conv = self.avgpool(conv)
        b, c, h, w = conv.size()
        conv = conv.view( b, w * h, c)
        x ,_= self.rnn(conv)
        x = self.fc(x)
        x = x.permute(1,0,2)
        return x


def test():
    md = CRNN_OCR()
    
    x = torch.randn(1, 3, 80, 160)
    # torch.onnx.export(md, x, 'sb.onnx', verbose=True)
    y = md(x)
    print(y.shape)
    # print(md)

test()