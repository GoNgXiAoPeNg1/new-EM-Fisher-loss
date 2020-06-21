import os
from math import ceil
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import pdb
import collections


def one_hot(scores, labels):
    _labels = torch.zeros_like(scores)
    _labels.scatter_(dim=1, index=labels, value=1)#scatter_  带下划线的函数 是inplace  ，就是内部赋值，没有返回值
    _labels.requires_grad = False
    return _labels



def jaccard(intersection, union, eps=1e-15):
    return 1 - (intersection + eps) / (union - intersection + eps)

def dice(intersection, union, eps=1e-15):
    return 1 - (2. * intersection + eps) / (union + eps)

def dice2(intersection, union, eps=1e-15):
#*F.sigmoid(union - 2.*intersection)
    return 1 - (2. * intersection + eps) / (20.*(union - 2.*intersection) + 2. * intersection + eps)

class BCESoftJaccardDice(nn.Module):

    def __init__(self, bce_weight=0., mode="dice", eps=1e-15, weight=None):
        super(BCESoftJaccardDice, self).__init__()
        self.nll_loss = torch.nn.BCEWithLogitsLoss(weight=weight)
        self.bce_weight = bce_weight
        self.eps = eps
        self.mode = mode

    def forward(self, outputs, targets):

        targets = targets.unsqueeze(1)
        targets = one_hot(outputs, targets)
        loss = self.bce_weight * self.nll_loss(outputs, targets)

        if self.bce_weight < 1.:
            targets = (targets == 1).float()
            outputs = torch.sigmoid(outputs)
            intersection = (outputs * targets).sum()
            union = outputs.sum() + targets.sum()
            if self.mode == "dice":
                score = dice(intersection, union, self.eps)
            elif self.mode == "jaccard":
                score = jaccard(intersection, union, self.eps)
            loss += (1 - self.bce_weight) * (score)
        return loss

class BCESoftJaccardDiceRateChange(nn.Module):

    def __init__(self, mode="dice", eps=1e-15, weight=None):
        super(BCESoftJaccardDiceRateChange, self).__init__()
        self.nll_loss = torch.nn.BCEWithLogitsLoss(weight=weight)
        self.eps = eps
        self.mode = mode


    def forward(self, outputs, targets):
        targets = targets.unsqueeze(1)
        targets = one_hot(outputs, targets)
       
        # targets = (targets == 1).float()
        outputs = torch.sigmoid(outputs)
        intersection = (outputs * targets).sum()
        union = outputs.sum() + targets.sum()

        loss = dice2(intersection, union, self.eps)

        return loss

def as_numpy(obj):
    if isinstance(obj, collections.Sequence):
        return [as_numpy(v) for v in obj]
    elif isinstance(obj, collections.Mapping):
        return {k: as_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, Variable):
        return obj.data.cpu().numpy()
    elif torch.is_tensor(obj):
        return obj.cpu().numpy()
    else:
        return np.array(obj)

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

 
def accuracy(y_pred, y_actual, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    _,pred = y_pred.topk(1, 1, True, True)
    batch_size = y_actual.size(0)
 
    pred = pred.squeeze()
    correct = len(pred[pred == y_actual])
    wrong = len(pred[pred != y_actual])
    return correct/(correct+wrong)

    return res

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):

        return self.nll_loss(F.log_softmax(inputs,dim=1), targets)


class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, weight=None, size_average=True, ignore_index=255):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss((1 - F.softmax(inputs,dim=1)) ** self.gamma * F.log_softmax(inputs,dim=1), targets)



class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, input, target, weights=None):

        C = target.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        dice = DiceLoss()
        totalLoss = 0

        for i in range(C):
            diceLoss = dice(input[:, i], target[:, i])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss

        return totalLoss
# MyLoss 是没有求均值，loss值很大，但是效果好，损失值降得快
class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.tanh = nn.ReLU(inplace=True)
        self.mse = nn.MSELoss()
        self.eps = 1e-15
    def forward(self, input, target):
        target = target.float()
        input = F.softmax(input, dim=1)
        neg_target = 1 - target

        pos1 = torch.pow(torch.sub(1, (torch.mul(input[:, 0], neg_target).sum() + self.eps)), 2)
        pos2 = torch.pow(torch.sub(1, (torch.mul(input[:, 1], target).sum() + self.eps)), 2)
        pos_loss = (pos1 + pos2) / 2

        neg1 = torch.pow(torch.mul(input[:, 0], target), 2)
        neg2 = torch.pow(torch.mul(input[:, 1], neg_target), 2)

        neg_loss = (neg1.sum() + neg2.sum()) / 2
        # if the follow  two lines are not used, the effect is better. But the loss value not in [0,1]
        # neg_loss = F.sigmoid(torch.log(torch.log(neg_loss+1)+1)+1)
        # pos_loss = F.sigmoid(pos_loss+1)

        
        return (pos_loss.sum() + neg_loss.sum()) / 2

#MyLossright的损失值小于1，在做科研时一般都要归一化到1.但是这个效果没有上边的好。所以在做工程时用上边的MyLoss
class MyLossright(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.tanh = nn.ReLU(inplace=True)
        self.mse = nn.MSELoss()
        self.eps = 1e-15
    def forward(self, input, target):
        target = target.float()
        input = F.softmax(input, dim=1)
        neg_target = 1 - target

        pos1 = torch.pow(torch.sub(1, (torch.mul(input[:, 0], neg_target).sum() + self.eps) / neg_target.sum()), 2)
        pos2 = torch.pow(torch.sub(1, (torch.mul(input[:, 1], target).sum() + self.eps) / target.sum()), 2)


        pos_loss = (pos1 + pos2) / 2

        neg1 = torch.pow(torch.mul(input[:, 0], target), 2)
        neg2 = torch.pow(torch.mul(input[:, 1], neg_target), 2)

        neg_loss = (neg1.sum() + neg2.sum()) / 2

        neg_loss = F.sigmoid(torch.log(torch.log(neg_loss+1)+1)+1)
        pos_loss = F.sigmoid(pos_loss+1)


        return (pos_loss.mean() + neg_loss.mean()) / 2

    
    #以下的损失效果不好。都是在做实验时用过的损失函数
    
class MyLossWithEdge(nn.Module):
    def __init__(self):
        super(MyLossWithEdge, self).__init__()
        self.eps = 1e-15
        # 通过设置核的大小确定求几个像素点边缘
        self.kernel_x = torch.unsqueeze(torch.nn.Parameter(torch.Tensor([[[-1,-1, -1, 0, 1, 1, 1],
                                                                          [-1, -1, -1, 0, 1, 1, 1],
                                                                          [-1, -1, -1, 0, 1, 1, 1],
                                                                          [-1, -1, -1, 0, 1, 1, 1],
                                                                          [-1, -1, -1, 0, 1, 1, 1],
                                                                          [-1, -1, -1, 0, 1, 1, 1],
                                                                          [-1, -1, -1, 0, 1, 1, 1]]]), requires_grad=False), 0).cuda()
        self.kernel_y = torch.unsqueeze(torch.nn.Parameter(torch.Tensor([[[-1, -1, -1, -1, -1, -1, -1],
                                                                          [-1, -1, -1, -1, -1, -1, -1],
                                                                          [-1, -1, -1, -1, -1, -1, -1],
                                                                          [0, 0, 0, 0, 0, 0, 0],
                                                                          [1, 1, 1, 1, 1, 1, 1],
                                                                          [1, 1, 1, 1, 1, 1, 1],
                                                                          [1, 1, 1, 1, 1, 1, 1]]]), requires_grad=False), 0).cuda()
        '''
        experience with kernal=3
        '''
        # self.kernel_x = torch.unsqueeze(torch.nn.Parameter(torch.Tensor([[[-1, 0, 1],
        #                                                                   [-2, 0, 2],
        #                                                                   [-1, 0, 1]]]),
        #                                                    requires_grad=False), 0).cuda()
        # self.kernel_y = torch.unsqueeze(torch.nn.Parameter(torch.Tensor([[
        #                                                                   [-1, -2, -1],
        #                                                                   [0, 0, 0],
        #                                                                   [1, 2, 1]]]),
        #                                                    requires_grad=False), 0).cuda()

    def forward(self, input, target):
        target = target.float()
        input = F.softmax(input, dim=1)
        neg_target = 1 - target
        x = F.conv2d(torch.unsqueeze(target.float(), 1), self.kernel_x, padding=3)
        y = F.conv2d(torch.unsqueeze(target.float(), 1), self.kernel_y, padding=3)
        target_edge = torch.pow((torch.pow(x, 2).float() + torch.pow(y, 2).float()).float() + self.eps, 0.5)
        target_edge = torch.squeeze(target_edge, dim=1)
        # target_edge = torch.where(target_edge<0.001, target_edge, torch.FloatTensor([0]).cuda())
        pos_target_edge = torch.mul(target, target_edge * 0.005)
        neg_target_edge = torch.mul(neg_target, target_edge * 0.2)
        #
        # pdb.set_trace()
        target = target + pos_target_edge
        neg_target = neg_target + neg_target_edge


        # cv2.imwrite('true1.png', target[0][0].cpu().numpy() * 25)
        pos1 = torch.pow(
            torch.sub(1, (torch.mul(input[:, 0], neg_target).sum(dim=1).sum(dim=1) + self.eps) / neg_target.sum(
                dim=1).sum(dim=1)), 2)
        pos2 = torch.pow(
            torch.sub(1, (torch.mul(input[:, 1], target).sum(dim=1).sum(dim=1) + self.eps) / target.sum(dim=1).sum(
                dim=1)), 2)
        pos_loss = (pos1 + pos2) / 2

        neg1 = torch.pow(torch.mul(input[:, 0], target), 2)
        neg2 = torch.pow(torch.mul(input[:, 1], neg_target), 2)
        neg_loss = (neg1.sum(dim=1).sum(dim=1) + neg2.sum(dim=1).sum(dim=1)) / 2

        neg_loss = F.sigmoid(torch.unsqueeze(torch.log(torch.log(neg_loss + 1) + 1) + 1, dim=0))
        pos_loss = F.sigmoid(torch.unsqueeze(pos_loss, dim=0))

        return (pos_loss.mean() + neg_loss.mean()) / 2

class MyLossWithEdgeAndDirctor(nn.Module):
    def __init__(self):
        super(MyLossWithEdgeAndDirctor, self).__init__()
        self.eps = 1e-15
        # 通过设置核的大小确定求几个像素点边缘

        '''
        experience with kernal=3
        '''
        self.kernel_x = torch.unsqueeze(torch.nn.Parameter(torch.Tensor([[[-1, 0, 1],
                                                                          [-2, 0, 2],
                                                                          [-1, 0, 1]]]),
                                                           requires_grad=False), 0).cuda()
        self.kernel_y = torch.unsqueeze(torch.nn.Parameter(torch.Tensor([[
                                                                          [-1, -2, -1],
                                                                          [0, 0, 0],
                                                                          [1, 2, 1]]]),
                                                           requires_grad=False), 0).cuda()

    def forward(self, input, target):
        target = target.float()
        input = F.softmax(input, dim=1)
        neg_target = 1 - target
        x_target = F.conv2d(torch.unsqueeze(target.float(), 1), self.kernel_x, padding=1)
        y_target = F.conv2d(torch.unsqueeze(target.float(), 1), self.kernel_y, padding=1)

        x_input = F.conv2d(torch.unsqueeze(input.float(), 1), self.kernel_x, padding=1)
        y_input = F.conv2d(torch.unsqueeze(input.float(), 1), self.kernel_y, padding=1)
        # the final number is used to augument the edge of target
        # target_edge = torch.pow((torch.pow(x_target, 2).float() + torch.pow(y_target, 2).float()).float() + self.eps, 0.5) * (0.005)
        # target_edge = torch.squeeze(target_edge, dim=1)

        # pos_target_edge = torch.mul(target, target_edge)
        # neg_target_edge = torch.mul(neg_target, target_edge)

        # target = target + pos_target_edge
        # neg_target = neg_target + neg_target_edge

        '''
        consider the director
        '''



        # cv2.imwrite('true1.png', target[0][0].cpu().numpy() * 25)
        pos1 = torch.div(1, torch.mul(input[:, 0], neg_target).sum() / neg_target.sum()) - 1
        pos2 = torch.div(1, torch.mul(input[:, 1], target).sum() / target.sum()) - 1
        pos_loss = (pos1 + pos2) / 2

        neg1 = torch.pow(torch.mul(input[:, 0], target), 2)
        neg2 = torch.pow(torch.mul(input[:, 1], neg_target), 2)
        neg_loss = (neg1.sum() + neg2.sum()) / 2

        neg_loss = F.sigmoid(torch.unsqueeze(torch.log(torch.log(neg_loss + 1) + 1) + 1, dim=0))
        pos_loss = F.sigmoid(torch.unsqueeze(pos_loss, dim=0))

        return (pos_loss.sum() + neg_loss.sum()) / 2


class MyLossNoSoftmax(nn.Module): # performence is very bad
    def __init__(self):
        super(MyLossNoSoftmax, self).__init__()
        self.tanh = nn.ReLU(inplace=True)
        self.mse = nn.MSELoss()
    def forward(self, input, target):
        target = target.float()
        neg_target = 1 - target

        pos1 = torch.div(1, torch.mul(input[:, 0], neg_target).sum() / neg_target.sum())-1
        pos2 = torch.div(1, torch.mul(input[:, 1], target).sum() / target.sum())-1
        pos_loss = (pos1 + pos2) / 2

        neg1 = torch.pow(torch.mul(input[:, 0], target), 2)
        neg2 = torch.pow(torch.mul(input[:, 1], neg_target), 2)
        neg_loss = (neg1.sum() + neg2.sum()) / 2

        neg_loss = F.sigmoid(torch.unsqueeze(torch.log(torch.log(neg_loss+1)+1)+1, dim=0))
        pos_loss = F.sigmoid(torch.unsqueeze(pos_loss, dim=0))

        return (pos_loss.sum() + neg_loss.sum()) / 2


class MyLossPixelIntraAndInter(nn.Module):
    def __init__(self):
        super(MyLossPixelIntraAndInter, self).__init__()
        self.tanh = nn.ReLU(inplace=True)
        self.mse = nn.MSELoss()
    def forward(self, input, target):
        target = target.float()
        input = F.softmax(input, dim=1)
        neg_target = 1 - target

        pos1_mean = torch.mul(input[:, 0], neg_target).sum() / neg_target.sum()
        pos1 = torch.pow(torch.sub(1, pos1_mean), 2)
        pos2_mean = torch.mul(input[:, 1], target).sum() / target.sum()
        pos2 = torch.pow(torch.sub(1, pos2_mean), 2)
        pos_loss = (pos1 + pos2) / 2

        pos1_var = torch.pow(torch.sub(input[:, 0] * neg_target, pos1_mean), 2).sum() - torch.pow(pos1_mean, 2) * target.sum()

        pos2_var = torch.pow(input[:, 1] * target - pos2_mean, 2).sum() - torch.pow(pos2_mean, 2) * neg_target.sum()

        pos_var = (pos1_var + pos2_var) / 2  #intra-class var
        pos_mean = (pos1_mean + pos2_mean) / 2 #intra-class center


        neg1_temp = torch.mul(input[:, 0], target)
        neg1_mean = neg1_temp.sum() / target.sum()
        neg1 = torch.pow(neg1_temp, 2)
        neg2_temp = torch.mul(input[:, 1], neg_target)
        neg2_mean = neg2_temp.sum() / neg_target.sum()
        neg2 = torch.pow(neg2_temp, 2)

        neg_loss = (neg1.sum() + neg2.sum()) / 2
        neg1_var = torch.pow(input[:, 0] * target - neg1_mean.sum(), 2) - torch.pow(neg1_mean, 2) * neg_target.sum()
        neg2_var = torch.pow(input[:, 1] * neg_target - neg2_mean, 2).sum() - torch.pow(neg2_mean, 2) * target.sum()
        neg_var = (neg1_var + neg2_var) / 2 #intra-class var
        neg_mean = (neg1_mean + neg2_mean) / 2 # intra-class center

        center_distance = torch.pow(pos_mean - neg_mean, 2)
        neg_center_distance = torch.log(torch.div(1, center_distance))
        neg_center_distance = F.sigmoid(torch.unsqueeze(neg_center_distance, dim=0))
        var = (pos_var + neg_var) / 2
        var = F.sigmoid(torch.unsqueeze(torch.log(torch.log(var + 1) + 1) + 1, dim=0))

        neg_loss = F.sigmoid(torch.unsqueeze(torch.log(torch.log(neg_loss+1)+1)+1, dim=0))
        pos_loss = F.sigmoid(torch.unsqueeze(pos_loss, dim=0))
        area = (pos_loss.mean() + neg_loss.mean()) / 2
        # return (pos_loss.mean() + neg_loss.mean() + neg_center_distance.mean() + var.mean()) / 4
        return (area + neg_center_distance.mean()) / 2


class MyLossVectorIntraAndInter(nn.Module):
    def __init__(self):
        super(MyLossVectorIntraAndInter, self).__init__()
        self.eps = 1e-15
        # 通过设置核的大小确定求几个像素点边缘
        self.kernel_x = torch.unsqueeze(torch.nn.Parameter(torch.Tensor([[[-1, -1, -1, 0, 1, 1, 1],
                                                                          [-1, -1, -1, 0, 1, 1, 1],
                                                                          [-1, -1, -1, 0, 1, 1, 1],
                                                                          [-1, -1, -1, 0, 1, 1, 1],
                                                                          [-1, -1, -1, 0, 1, 1, 1],
                                                                          [-1, -1, -1, 0, 1, 1, 1],
                                                                          [-1, -1, -1, 0, 1, 1, 1]]]),
                                                           requires_grad=False), 0).cuda()
        self.kernel_y = torch.unsqueeze(torch.nn.Parameter(torch.Tensor([[[-1, -1, -1, -1, -1, -1, -1],
                                                                          [-1, -1, -1, -1, -1, -1, -1],
                                                                          [-1, -1, -1, -1, -1, -1, -1],
                                                                          [0, 0, 0, 0, 0, 0, 0],
                                                                          [1, 1, 1, 1, 1, 1, 1],
                                                                          [1, 1, 1, 1, 1, 1, 1],
                                                                          [1, 1, 1, 1, 1, 1, 1]]]),
                                                           requires_grad=False), 0).cuda()
        self.tanh = nn.ReLU(inplace=True)
        self.mse = nn.MSELoss()
    def forward(self, input, target):
        target = target.float()
        input = F.softmax(input, dim=1)
        neg_target = 1 - target

        x = F.conv2d(torch.unsqueeze(target.float(), 1), self.kernel_x, padding=3)
        y = F.conv2d(torch.unsqueeze(target.float(), 1), self.kernel_y, padding=3)
        target_edge = torch.pow((torch.pow(x, 2).float() + torch.pow(y, 2).float()).float() + self.eps, 0.5)
        target_edge = torch.squeeze(target_edge, dim=1)
        pos_target_edge = torch.mul(target, target_edge * 0.005)
        neg_target_edge = torch.mul(neg_target, target_edge * 0.2)
        pos_target_edge = target + pos_target_edge
        neg_target_edge = neg_target + neg_target_edge

        pos1_mean = torch.mul(input[:, 0], neg_target).sum() / neg_target.sum()
        pos1 = torch.div(1, torch.mul(input[:, 0], neg_target_edge).sum() / neg_target_edge.sum()) - 1
        pos2_mean = torch.mul(input[:, 1], target).sum() / target.sum()
        pos2 = torch.div(1, torch.mul(input[:, 1], pos_target_edge).sum() / pos_target_edge.sum()) - 1
        pos_loss = (pos1 + pos2) / 2
        pos1_var = torch.pow(1 - (input[:, 0] * neg_target).sum() / neg_target.sum(), 2)

        pos2_var = torch.pow(1 - (input[:, 1] * target).sum() / target.sum(), 2)

        pos_var = (pos1_var + pos2_var) / 2  # intra-class var

        neg1_temp = torch.mul(input[:, 0], target)
        neg1_mean = neg1_temp.sum() / target.sum()
        neg1 = torch.pow(torch.mul(input[:, 0], pos_target_edge), 2).sum() / target.sum()
        neg2_temp = torch.mul(input[:, 1], neg_target)
        neg2_mean = neg2_temp.sum() / neg_target.sum()
        neg2 = torch.pow(torch.mul(input[:, 1], neg_target_edge), 2).sum() / neg_target.sum()
        neg_loss = (neg1.sum() + neg2.sum()) / 2
        neg1_var = (input[:, 0] * target).sum() / target.sum()
        neg2_var = (input[:, 1] * neg_target).sum() / neg_target.sum()
        neg_var = (neg1_var + neg2_var) / 2  # intra-class var

        center_distance = torch.pow(pos1_mean - neg1_mean, 2) + torch.pow(neg2_mean - pos2_mean, 2)
        neg_center_distance = torch.pow(2 - center_distance, 2)
        neg_center_distance = torch.sigmoid(neg_center_distance) - 0.5

        var = (pos_var + neg_var) / 2
        var = F.sigmoid(var) - 0.5

        neg_loss = F.sigmoid(torch.unsqueeze(neg_loss, dim=0))-0.5
        pos_loss = F.sigmoid(torch.unsqueeze(pos_loss, dim=0))-0.5
        area = (neg_loss + pos_loss) / 2

        f = open('/media/gongxp/2eae26dd-69c7-4195-94af-3604072e182f/gongxp/model_path/new3.log',mode='a')
        f.writelines(str((area.cpu().detach().numpy())[0]) + '      ' +
        str((neg_center_distance.cpu().detach().numpy())) + '      ' + str((var.cpu().detach().numpy()))+'\n')
        f.close()
        print((str((area.cpu().detach().numpy())[0]) + '      ' +
        str((neg_center_distance.cpu().detach().numpy())) + '      ' + str((var.cpu().detach().numpy()))))

        return (area + neg_center_distance.mean() + var.mean()) / 3
        # return area
        #

        # return 0.2*(pos_loss.mean() + neg_loss.mean()) + 0.3*neg_center_distance.mean() + 0.5*var.mean()

class MyLossPixelIntraAndInterAndMaxMin(nn.Module):
    def __init__(self):
        super(MyLossPixelIntraAndInterAndMaxMin, self).__init__()
        self.tanh = nn.ReLU(inplace=True)
        self.mse = nn.MSELoss()
    def forward(self, input, target):
        target = target.float()
        input = F.softmax(input, dim=1)
        neg_target = 1 - target

        pos1_mean = torch.mul(input[:, 0], neg_target).sum() / neg_target.sum()
        pos1 = torch.div(1, pos1_mean) - 1
        pos2_mean = torch.mul(input[:, 1], target).sum() / target.sum()
        pos2 = torch.div(1, pos2_mean) - 1
        pos_loss = (pos1 + pos2) / 2

        pos1_var = torch.pow(torch.sub(input[:, 0] * neg_target, pos1_mean), 2).sum() - torch.pow(pos1_mean, 2) * target.sum()
        pos2_var = torch.pow(input[:, 1] * target - pos2_mean, 2).sum() - torch.pow(pos2_mean, 2) * neg_target.sum()

        pos_var = (pos1_var + pos2_var) / 2  #intra-class var
        pos_mean = (pos1_mean + pos2_mean) / 2 #intra-class center

        Min = torch.cat([torch.unsqueeze(torch.mul(input[:, 0], neg_target).min(dim=1)[0].min(dim=1)[0], dim=1), torch.unsqueeze(torch.mul(input[:, 1], target).min(dim=1)[0].min(dim=1)[0], dim=1)], dim=1).min(dim=1)[0]
        Max = torch.cat([torch.unsqueeze(torch.mul(input[:, 0], target).max(dim=1)[0].max(dim=1)[0], dim=1), torch.unsqueeze(torch.mul(input[:, 1], neg_target).max(dim=1)[0].max(dim=1)[0], dim=1)], dim=1).max(dim=1)[0]
        neg_MaxMin_distance = torch.sigmoid(torch.div(1, torch.pow(Min - Max, 2))+1)

        neg1_temp = torch.mul(input[:, 0], target)
        neg1_mean = neg1_temp.sum() / target.sum()
        neg1 = torch.pow(neg1_temp, 2)
        neg2_temp = torch.mul(input[:, 1], neg_target)
        neg2_mean = neg2_temp.sum() / neg_target.sum()
        neg2 = torch.pow(neg2_temp, 2)

        neg_loss = (neg1.sum() + neg2.sum()) / 2
        neg1_var = torch.pow(input[:, 0] * target - neg1_mean, 2).sum() - torch.pow(neg1_mean, 2) * neg_target.sum()
        neg2_var = torch.pow(input[:, 1] * neg_target - neg2_mean, 2).sum() - torch.pow(neg2_mean, 2) * target.sum()
        neg_var = (neg1_var + neg2_var) / 2 # intra-class var
        neg_mean = (neg1_mean + neg2_mean) / 2 # intra-class center

        center_distance = torch.pow(pos_mean - neg_mean, 2)
        neg_center_distance = torch.log(torch.div(1, center_distance))
        neg_center_distance = F.sigmoid(torch.unsqueeze(neg_center_distance, dim=0))
        var = (pos_var + neg_var) / 2
        var = F.sigmoid(torch.unsqueeze(torch.log(torch.log(var + 1) + 1) + 1, dim=0))

        neg_loss = F.sigmoid(torch.unsqueeze(torch.log(torch.log(neg_loss+1)+1)+1, dim=0))
        pos_loss = F.sigmoid(torch.unsqueeze(pos_loss, dim=0)+1)

        return (pos_loss.mean() + neg_loss.mean() + neg_center_distance.mean() + var.mean() + neg_MaxMin_distance.mean()) / 5
