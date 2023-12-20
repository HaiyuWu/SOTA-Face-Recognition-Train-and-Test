import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class ArcFace(nn.Module):
    def __init__(self, classnum, num_features=512, s=64.0, m=0.50):
        super(ArcFace, self).__init__()
        self.num_features = num_features
        self.n_classes = classnum
        self.s = s
        self.m = m
        self.W = Parameter(torch.FloatTensor(classnum, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, embeddings, label=None):
        # normalize features
        x = F.normalize(embeddings)
        # normalize weights
        W = F.normalize(self.W)
        # cosine similarity of x, W
        logits = F.linear(x, W)
        if label is None:
            return logits
        # theta: distance between x an W
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        # Add margin to enlarge the distance
        label_logits = torch.cos(theta + self.m)
        #
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + label_logits * one_hot
        # feature re-scale
        output *= self.s

        return output


class SphereFace(nn.Module):
    def __init__(self, classnum, num_features=512, s=64.0, m=1.35):
        super(SphereFace, self).__init__()
        self.num_features = num_features
        self.n_classes = classnum
        self.s = s
        self.m = m
        self.W = Parameter(torch.FloatTensor(classnum, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, embeddings, label=None):
        # normalize features
        x = F.normalize(embeddings)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # add margin
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        label_logits = torch.cos(self.m * theta)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + label_logits * one_hot
        # feature re-scale
        output *= self.s

        return output


class CosFace(nn.Module):
    def __init__(self, classnum, num_features=512, s=64.0, m=0.35):
        super(CosFace, self).__init__()
        self.num_features = num_features
        self.n_classes = classnum
        self.s = s
        self.m = m
        self.W = Parameter(torch.FloatTensor(classnum, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, embeddings, label=None):
        # normalize features
        x = F.normalize(embeddings)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # add margin
        label_logits = logits - self.m
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + label_logits * one_hot
        # feature re-scale
        output *= self.s

        return output


##############
#   AdaFace  #
##############
def l2_norm(embeddings,axis=1):
    norm = torch.norm(embeddings,2,axis,True)
    output = torch.div(embeddings, norm)
    return output


class AdaFace(nn.Module):
    def __init__(self,
                 classnum,
                 num_features=512,
                 m=0.4,
                 h=0.333,
                 s=64.,
                 t_alpha=1.0,
                 ):
        super(AdaFace, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(num_features, classnum))

        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m
        self.eps = 1e-3
        self.h = h
        self.s = s

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1)*(20))
        self.register_buffer('batch_std', torch.ones(1)*100)

    def forward(self, embeddings, label=None):
        norms = torch.norm(embeddings, 2, 1, True)
        embeddings = torch.div(embeddings, norms)

        kernel_norm = l2_norm(self.kernel, axis=0)
        cosine = torch.mm(embeddings, kernel_norm)
        if label is None:
            return cosine
        cosine = cosine.clamp(-1+self.eps, 1-self.eps) # for stability

        safe_norms = torch.clip(norms, min=0.001, max=100) # for stability
        safe_norms = safe_norms.clone().detach()

        # update batchmean batchstd
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std+self.eps) # 66% between -1, 1
        margin_scaler = margin_scaler * self.h # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1)

        # g_angular
        m_arc = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)
        g_angular = self.m * margin_scaler * -1
        m_arc = m_arc * g_angular
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi-self.eps)
        cosine = theta_m.cos()

        # g_additive
        m_cos = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        cosine = cosine - m_cos

        # scale
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m


###############
# Circle Loss #
###############
def convert_label_to_similarity(feature, label):
    normed_feature = F.normalize(feature)
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


class CircleLoss(nn.Module):
    def __init__(self, classnum, m=0.25, gamma=256) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma

    def forward(self, embeddings, label=None):
        if label is None:
            return embeddings
        sp, sn = convert_label_to_similarity(embeddings, label)
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        logit = torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0)

        return logit
