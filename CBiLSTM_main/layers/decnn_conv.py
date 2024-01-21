import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DECNN_CONV(nn.Module):
    #input_dim: 200
    def __init__(self, input_dim, opt):
        super(DECNN_CONV, self).__init__()

        self.conv1 = torch.nn.Conv1d(input_dim, 128, 5, padding=2)
        self.conv2 = torch.nn.Conv1d(input_dim, 128, 3, padding=1)
        self.dropout = torch.nn.Dropout(opt.keep_prob)

        self.conv3 = torch.nn.Conv1d(256, 256, 5, padding=2)
        self.conv4 = torch.nn.Conv1d(256, 256, 5, padding=2)
        self.conv5 = torch.nn.Conv1d(256, 256, 5, padding=2)


        # 双向的lstm
        self.lstm = nn.LSTM(110,opt.hidden_dim,bidirectional=True,batch_first=True)
        self.linear = nn.Linear(512,110)
        self.dropout = torch.nn.Dropout(opt.keep_prob)
        #BiRNN
        self.rnn = nn.RNN(110, opt.hidden_dim, bidirectional=True)
        self.linear = nn.Linear(512, 110)
        self.dropout = torch.nn.Dropout(opt.keep_prob)

    def forward(self, inputs): #inputs torch.Size([8, 200, 110])
        x_conv = torch.nn.functional.relu(torch.cat((self.conv1(inputs), self.conv2(inputs)), dim=1))
        x_conv = self.dropout(x_conv)
        x_conv = torch.nn.functional.relu(self.conv3(x_conv))
        x_conv = self.dropout(x_conv)
        x_conv = torch.nn.functional.relu(self.conv4(x_conv))
        x_conv = self.dropout(x_conv)
        x_conv = torch.nn.functional.relu(self.conv5(x_conv))
        #x_conv: torch.Size([8, 110, 256])


        #Bilstm
        x_trans,_ = self.lstm(inputs)
        x_trans = self.linear(x_trans.reshape(-1,512)).reshape(inputs.size(0),inputs.size(1),-1)
        x_trans = x_trans.unsqueeze(0)
        x_trans = F.interpolate(x_trans,size=(256,110),mode="nearest")
        x_trans = torch.nn.functional.relu(x_trans.squeeze(0)) #torch.Size([8, 256, 110])
        x_trans = self.dropout(x_trans)



        # x_conv = torch.mean(torch.stack((x_conv,x_birnn,x_trans)),dim=0) #x: torch.Size([8, 110, 512])
        x = torch.add(x_trans,x_conv)
        # print(x_conv.shape)
        x_conv = x_conv.transpose(1, 2)
        return x_conv