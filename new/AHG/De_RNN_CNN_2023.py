import torch.nn as nn
import torch.nn.functional as F
import torch
class CNN_decoder(nn.Module):
    def __init__(self,node_num): #10  768  37133
        super(CNN_decoder, self).__init__()
        self.node_num=node_num

        self.conv1 = nn.Sequential(  # 1*节点数*维度
            nn.Conv1d(
                in_channels=self.node_num,  # input height  gray just have one level
                out_channels=10,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
                # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(10, 10, 5, 1, 2),  #
            nn.ReLU(),  # activation
            nn.MaxPool2d(1),  # output shape
        )

        self.l1 = nn.Linear(10 * 768, 10 * self.node_num)
        
    def forward(self, x1): #1*节点数*嵌入大小    1*480*768  test:torch.Size([1, 14362, 768])
        cc = x1.cuda().data.cpu().numpy().tolist() #1*480
        numm=torch.Tensor(len(cc[0])).cuda()
        self.node_num=numm #480
        self.conv1 = nn.Sequential(nn.Conv1d(len(cc[0]),  10, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(kernel_size=1), )
        self.conv1=self.conv1.cuda() #480 10
        self.l1 = nn.Linear(10 * 768, 10 * len(cc[0]))
        self.l1 = self.l1.cuda()  # 480 10
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)  #torch.Size([1,10, 768])

        x =x1.reshape(1, -1)  # 1*(10*768) 1*7680
        x = x.squeeze() #7680
        x = self.l1(x)  ##143620
        x = x.relu()  #143620
        x = F.dropout(x, p=0.2)  # 长度为10的张量  应该是10*vocab

        x = x.reshape(10, -1) #10*vocab  test:10*143262
        x = F.softmax(x, dim=1) #10
        x = torch.argmax(x, dim=1)  # 10个id
        return x1,x  #10个词的嵌入torch.Size([1, 10, 768])


