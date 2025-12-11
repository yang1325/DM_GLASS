import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)

        alpha = self.alpha
        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def forward(self, z_i, z_j):
        """
        z_i, z_j: (B, D) 兩個視角的特徵向量
        """
        bs = z_i.shape[0]

        z = torch.cat([z_i, z_j], dim=0)
        z_norm = F.normalize(z, dim=1)
        sim_matrix = torch.matmul(z_norm, z_norm.T) / self.temperature

        mask = torch.eye(2 * bs, dtype=torch.bool).to(z_i.device)
        sim_matrix = sim_matrix.masked_fill(mask, -9e15)
        
        # 4. 計算 Loss
        pos_mask = torch.zeros((2 * bs, 2 * bs), dtype=torch.bool).to(z_i.device)
        pos_mask.fill_diagonal_(False)
        for i in range(bs):
            pos_mask[i, i + bs] = True
            pos_mask[i + bs, i] = True
            
        # 選出正樣本對的分數
        # (因為 SimCLR 比較複雜，這裡可以用簡化的 MSE 或 Cosine Loss 替代，如果是 5-shot)
        
        # --- [5-shot 簡易版 Contrastive Loss] ---
        # 因為 Batch Size 很小，標準 SimCLR 需要大 Batch 才有足夠負樣本。
        # 在 5-shot 下，建議使用 "SimSiam" 風格或純粹的 "Cosine Embedding Loss"
        # 目標：最大化 z_i 和 z_j 的相似度
        
        loss = 1 - self.cosine_similarity(z_i, z_j).mean()
        return loss