import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
from math import ceil, floor

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()

class Model(nn.Module):
    def __init__(self, num_class, s, omega):
        super(Model, self).__init__()

        self.num_class = num_class
        self.s = s
        self.omega = omega

        D = 1024
        d = 0.7

        self.fc_r = nn.Linear(D, D)
        self.fc1_r = nn.Linear(D, D)
        self.fc_f = nn.Linear(D, D)
        self.fc1_f = nn.Linear(D, D)
        self.classifier_r = nn.Conv1d(D, num_class, kernel_size=1)
        self.classifier_f = nn.Conv1d(D, num_class, kernel_size=1)
        self.classifier_ra = nn.ModuleList([nn.Conv1d(D, 1, kernel_size=1) for i in range(num_class)]) # it can be implemented by conv2d with groups=num_class
        self.classifier_fa = nn.ModuleList([nn.Conv1d(D, 1, kernel_size=1) for i in range(num_class)])

        self.dropout_r = nn.Dropout(d)
        self.dropout_f = nn.Dropout(d)

        self.apply(weights_init)

        self.mul_r = nn.Parameter(data=torch.ones(num_class))
        self.mul_f = nn.Parameter(data=torch.ones(num_class))

    def forward(self, inputs):
        N, T, D = inputs.shape  # 32,200,2048
        D //= 2
        x_r = F.relu(self.fc_r(inputs[:,:,:D]))
        x_f = F.relu(self.fc_f(inputs[:,:,D:]))
        x_r = F.relu(self.fc1_r(x_r)).permute(0,2,1)  # (32,1024,200)
        x_f = F.relu(self.fc1_f(x_f)).permute(0,2,1)  # (32,1024,200)

        x_r = self.dropout_r(x_r)
        x_f = self.dropout_f(x_f)

        k = max(T-floor(T/self.s), 1)  # 195
        cls_x_r = self.classifier_r(x_r).permute(0,2,1)   # (32,200,101)
        cls_x_f = self.classifier_f(x_f).permute(0,2,1)   # (32,200,101)
        cls_x_ra = cls_x_r.new_zeros(cls_x_r.shape)  # (32,200,101)   擦出了动作的非动作部分的得分
        cls_x_fa = cls_x_f.new_zeros(cls_x_f.shape)  # (32,200,101)
        cls_x_rat = cls_x_r.new_zeros(cls_x_r.shape) # (32,200,101)
        cls_x_fat = cls_x_f.new_zeros(cls_x_f.shape) # (32,200,101)

        mask_value = -100

        for i in range(self.num_class):
            """
            torch.kthvalue(input, k, dim=None, out=None) -> (Tensor, LongTensor):取输入张量input指定维度上第k个最小值，若不指定dim，
            则默认为input的最后一维。返回一个元组，其中indices是原始输入张量input中沿dim维的第k个最小值下标。
               input(Tensor) - 输入张量
               k(int) - 第k个最小值
               dim(int, optional)` - 沿着此维度进行排序
               out(tuple, optional) - 输出元组
            """
            mask_r = cls_x_r[:,:,i]>torch.kthvalue(cls_x_r[:,:,i], k, dim=1, keepdim=True)[0]   # (32,200)   也就是将每一个类别的特征前195时间维度的特征置为0
            x_r_erased = torch.masked_fill(x_r, mask_r.unsqueeze(1), 0)  # (32,1024,200)  非动作区域的特征，屏蔽了动作部分的特征
            # masked_fill_(mask, value)用value填充 self tensor 中的元素, 当对应位置的 mask 是1.
            cls_x_ra[:,:,i] = torch.masked_fill(self.classifier_ra[i](x_r_erased).squeeze(1), mask_r, mask_value)  # (32,200,101)
            cls_x_rat[:,:,i] = self.classifier_ra[i](x_r).squeeze(1)  # (32,200,101)    和cls_x_r有什么区别 

            mask_f = cls_x_f[:,:,i]>torch.kthvalue(cls_x_f[:,:,i], k, dim=1, keepdim=True)[0]  # (32,200)
            x_f_erased = torch.masked_fill(x_f, mask_f.unsqueeze(1), 0)   # (32,1024,200)
            cls_x_fa[:,:,i] = torch.masked_fill(self.classifier_fa[i](x_f_erased).squeeze(1), mask_f, mask_value)  # (32,200,101)
            cls_x_fat[:,:,i] = self.classifier_fa[i](x_f).squeeze(1)   # (32,200,101)

        tcam = (cls_x_r+cls_x_rat*self.omega) * self.mul_r + (cls_x_f+cls_x_fat*self.omega) * self.mul_f  # # (32,200,101)

        # (32,200,1024) [(32,200,101),(32,200,101)], (32,200,1024) [(32,200,101),(32,200,101)],  (32,200,101)
        return x_r.permute(0,2,1), [cls_x_r, cls_x_ra], x_f.permute(0,2,1), [cls_x_f, cls_x_fa], tcam
