import math
import torch
import torch.nn as nn
from torch.nn import Parameter


class CombinedMarginLoss(nn.Module):
    def __init__(self,
                 s=64,
                 m1=1.0,
                 m2=0.0,
                 m3=0.4,
                 interclass_filtering_threshold=0):
        super().__init__()
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.interclass_filtering_threshold = interclass_filtering_threshold

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        index_positive = torch.where(labels != -1)[0]

        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = logits > self.interclass_filtering_threshold
                dirty = dirty.float()
                mask = torch.ones([index_positive.size(0), logits.size(1)], device=logits.device)
                mask.scatter_(1, labels[index_positive], 0)
                dirty[index_positive] *= mask
                tensor_mul = 1 - dirty
            logits = tensor_mul * logits

        target_logit = logits[index_positive, labels[index_positive].view(-1)]

        if self.m1 == 1.0 and self.m3 == 0.0:
            with torch.no_grad():
                target_logit.arccos_()
                logits.arccos_()
                final_target_logit = target_logit + self.m2
                logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
                logits.cos_()
            logits *= self.s

        elif self.m3 > 0:
            final_target_logit = target_logit - self.m3
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            logits *= self.s
        else:
            raise

        return logits, None


class ArcFace(nn.Module):
    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.margin = margin

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        logits = logits.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        # index of the ground truth labels in the current batch
        index = torch.where(labels != -1)[0]
        # find the corresponding logit
        target_logit = logits[index, labels[index].view(-1)]
        with torch.no_grad():
            # convert to theta
            target_logit.arccos_()
            logits.arccos_()
            # add margin to target class
            final_target_logit = target_logit + self.margin
            # assign the added logit back to logits
            logits[index, labels[index].view(-1)] = final_target_logit
            # convert it back to cosine
            logits.cos_()
        # scale with pre-set value
        logits *= self.s
        return logits, None


class CurricularFace(nn.Module):
    def __init__(self, margin=0.5, s=64.):
        super(CurricularFace, self).__init__()
        self.margin = margin
        self.s = s
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.register_buffer('t', torch.zeros(1))

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        index = torch.where(labels != -1)[0]
        cos_theta = logits.clamp(-1, 1)
        target_chunk = cos_theta[index, :]
        target_logit = cos_theta[index, labels[index].view(-1)]

        labels = labels.view(-1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m #cos(target+margin)
        # find the place where the value of incorrect predictions are larger than correct prediction + margin -> hard samples
        mask = target_chunk > cos_theta_m[:, None]

        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = target_chunk[mask]
        with torch.no_grad():
            self.t = target_chunk.mean() * 0.01 + (1 - 0.01) * self.t

        target_chunk[mask] = hard_example * (self.t + hard_example)
        cos_theta[index, :] = target_chunk
        cos_theta[index, labels[index].view(-1)] = final_target_logit
        output = cos_theta * self.s
        return output, None


class SphereFace(nn.Module):
    def __init__(self, s=64.0, margin=1.7):
        super(SphereFace, self).__init__()
        self.s = s
        self.margin = margin

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        logits = logits.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]
        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            # sphereface multiply
            final_target_logit = target_logit * self.margin
            logits[index, labels[index].view(-1)] = final_target_logit
            logits.cos_()
        logits *= self.s

        return logits, None


class CosFace(nn.Module):
    def __init__(self, s=64.0, margin=0.4):
        super(CosFace, self).__init__()
        self.s = s
        self.margin = margin

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        logits = logits.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]
        with torch.no_grad():
            # cosface
            final_target_logit = target_logit - self.margin
            logits[index, labels[index].view(-1)] = final_target_logit
        logits *= self.s
        return logits, None


class AdaFace(nn.Module):
    def __init__(self,
                 margin=0.4,
                 h=0.333,
                 s=64.,
                 t_alpha=1.0,
                 ):
        super(AdaFace, self).__init__()
        self.margin = margin
        self.eps = 1e-3
        self.h = h
        self.s = s

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1)*(20))
        self.register_buffer('batch_std', torch.ones(1)*100)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        index = torch.where(labels != -1)[0]
        cosine = logits.clamp(-1+self.eps, 1-self.eps)
        target_logit = cosine[index, labels[index].view(-1)]

        norms = torch.norm(embeddings, 2, 1, True)
        safe_norms = torch.clip(norms, min=0.001, max=100)
        safe_norms = safe_norms[index].clone().detach()

        # update batchmean batchstd
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std = std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std+self.eps) # 66% between -1, 1
        margin_scaler = margin_scaler * self.h # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1).view(-1)

        # g_angular
        m_arc = target_logit.arccos_()
        g_angular = self.margin * margin_scaler * -1
        theta = cosine.arccos_()
        theta[index, labels[index].view(-1)] = m_arc * (1 + g_angular)
        theta_m = torch.clip(theta, min=self.eps, max=math.pi-self.eps)
        cosine = theta_m.cos()
        # g_additive
        g_add = self.margin + (self.margin * margin_scaler)
        cosine[index, labels[index].view(-1)] = cosine[index, labels[index].view(-1)] - g_add
        # scale
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m, None


class MagFace(nn.Module):
    def __init__(self, s=64.0, l_a=10, u_a=110, l_margin=0.45, u_margin=0.8):
        super(MagFace, self).__init__()
        self.s = s
        self.l_margin = l_margin
        self.u_margin = u_margin
        self.l_a = l_a
        self.u_a = u_a

    def _margin(self, x):
        margin = (self.u_margin - self.l_margin) / \
                 (self.u_a - self.l_a) * (x - self.l_a) + self.l_margin
        return margin

    def calc_loss_G(self, x_norm):
        g = 1 / (self.u_a ** 2) * x_norm + 1 / (x_norm)
        return torch.mean(g)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        x_norm = torch.norm(kwargs['embeddings'], dim=1, keepdim=True).clamp(self.l_a, self.u_a)[index]

        ada_margin = self._margin(x_norm)
        cos_m, sin_m = torch.cos(ada_margin).view(-1), torch.sin(ada_margin).view(-1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        # add margin
        logits[index, labels[index].view(-1)] = target_logit * cos_m - sin_theta * sin_m
        loss_g = self.calc_loss_G(x_norm)
        logits *= self.s

        return logits, loss_g


class UniFace(nn.Module):
    def __init__(self, margin=0.4, s=64, l=1.0, r=1.0, bias_init=15):
        super(UniFace, self).__init__()
        self.margin = margin
        self.s = s
        self.l = l
        self.r = r
        self.bias = Parameter(torch.FloatTensor(1))
        nn.init.constant_(self.bias, bias_init)

    def forward(self, cos_theta: torch.Tensor, labels: torch.Tensor, **kwargs):
        index = torch.where(labels != -1)[0]
        target_chunk = cos_theta[index, :]
        cos_m_theta_p = self.s * (target_chunk - self.margin) - self.bias
        cos_m_theta_n = self.s * target_chunk - self.bias

        p_loss = torch.log(1 + torch.exp(-cos_m_theta_p.clamp(min=-self.s, max=self.s)))
        n_loss = torch.log(1 + torch.exp(cos_m_theta_n.clamp(min=-self.s, max=self.s))) * self.l

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros((labels[index].size(0), kwargs['out_features']), dtype=torch.bool)
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, labels[index].view(-1, 1).long(), 1)
        loss = one_hot * p_loss + (~one_hot) * n_loss

        return loss.sum(dim=1).mean(), "uniface"


###############
# Circle Loss #
###############
def convert_label_to_similarity(normed_feature, label):
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


class CircleLoss(nn.Module):
    def __init__(self, m=0.25, gamma=256) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, label: torch.Tensor, norms: torch.Tensor):
        sp, sn = convert_label_to_similarity(norms, label)
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        logit = torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0)

        return logit
