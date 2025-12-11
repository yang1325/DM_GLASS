import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def init_weight(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    if isinstance(m, torch.nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, torch.nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)

class CoresetSampler:
    def __init__(self, percentage=0.1, device='cpu'):
        """
        Args:
            percentage: 取樣比例 (例如 0.1 代表取 10% 的特徵)
            device: 運算裝置
        """
        self.percentage = percentage
        self.device = device

    def run(self, features):
        """
        執行 Coreset Sampling
        Args:
            features: 原始特徵 (N, C) N=總Patch數, C=特徵維度
        Returns:
            sampled_features: 取樣後的特徵 (K, C)
        """
        if self.percentage >= 1.0 or len(features) == 0:
            return features

        n_samples = int(len(features) * self.percentage)
        if n_samples < 10: 
            return features
        
        # 1. JL 隨機投影
        with torch.no_grad():
            current_dim = features.shape[1]
            target_dim = 128
            
            # 建立隨機投影矩陣 (C, 128) 來隨機投影
            projection_matrix = torch.randn(current_dim, target_dim).to(self.device)
            projected_features = torch.matmul(features, projection_matrix)

        # 2. 從投影的特徵挑選重要的
        select_indices = self.coreset_sampling(projected_features, n_samples)
        
        # 3. 回傳原始的高維特徵
        return features[select_indices]

    def coreset_sampling(self, features, n_samples):
        """
        執行貪婪採樣邏輯
        """
        number_of_starting_points = 10
        n = features.shape[0]
        selected_indices = []
        min_distances = torch.full((n,), float('inf'), device=self.device)
        
        # 1. 隨機選幾個點作為起始
        current_selection = torch.randint(0, n, (number_of_starting_points,), device=self.device)
        selected_indices.extend(current_selection.tolist())
        dists = torch.cdist(features, features[current_selection])
        min_distances = torch.min(min_distances, dists.min(dim=1)[0])

        for _ in tqdm(range(n_samples - number_of_starting_points), desc="Coreset Sampling", leave=False):
            # 找出離代表點最遠的那個點
            new_index = torch.argmax(min_distances).item()
            selected_indices.append(new_index)
            
            # 更新距離表
            new_dist = torch.cdist(features, features[new_index].unsqueeze(0)).squeeze()
            min_distances = torch.min(min_distances, new_dist)
            
        return torch.tensor(selected_indices)

class SampledGCM(nn.Module):
    """
    跟memory bank裡面的所有特徵比，都不像的特徵，異常值就會高
    """
    def __init__(self, use_product_aggregation=False):
        super(SampledGCM, self).__init__()
        self.use_product = use_product_aggregation

    def forward(self, query_feat, memory_bank):
        """
        Args:
            query_feat: 當前輸入特徵 (B, C, H, W)
            memory_bank: 採樣後的特徵庫 (K, C)
        Returns:
            anomaly_map: (B, 1, H, W)
        """
        b, c, h, w = query_feat.size()
        
        # 1. 準備 Query: (B, C, H, W) -> (B, H*W, C)
        query_flat = query_feat.permute(0, 2, 3, 1).reshape(b, h*w, c)
        
        # 2. 特徵標準化 (Cosine Similarity 需要)
        query_norm = F.normalize(query_flat, p=2, dim=2) # (B, HW, C)
        mem_norm = F.normalize(memory_bank, p=2, dim=1)  # (K, C)

        # 3. 計算相似度
        sim_matrix = torch.matmul(query_norm, mem_norm.t())
        
        # 4. 取最相似的值
        # 如果容易誤判（把異常當正常），可以嘗試把 max 改成 topk(...).mean()
        values, _ = torch.max(sim_matrix, dim=2)
        
        # 5. Reshape
        commonality_map = values.reshape(b, 1, h, w)
        
        # 6. 反轉 (相似度高 -> 異常度低)
        anomaly_map = 1.0 - commonality_map
        
        return anomaly_map
    
class SpatialGCM(nn.Module):
    def __init__(self, kernel_size=3, use_product_aggregation=False):
        """
        Modified Group Co-attention Module for Anomaly Detection.
        
        Args:
            kernel_size (int): 局部搜尋的窗口大小 (e.g., 3 or 5)。
            use_product_aggregation (bool): 
                False (建議): 使用 Mean 聚合。對瑕疵檢測較穩定，只要與部分正常樣本相似即可。
                True (原版 GCM): 使用 Product 連乘。非常嚴格，要求與'所有'記憶庫樣本都相似 (容易過殺)。
        """
        super(SpatialGCM, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.use_product = use_product_aggregation

    def forward(self, query_feat, memory_bank):
        """
        Args:
            query_feat: 當前輸入特徵 (B, C, H, W)
            memory_bank: 正常樣本特徵庫 (M, C, H, W)
        
        Returns:
            anomaly_map: 異常注意力圖 (B, 1, H, W), 值域 [0, 1], 越高代表越異常
        """
        b, c, h, w = query_feat.size()
        m = memory_bank.size(0)

        # 1. 特徵標準化 (Cosine Similarity 前置作業)
        # 沿著 Channel 維度做 L2 Normalize
        q_norm = F.normalize(query_feat, p=2, dim=1)      # (B, C, H, W)
        mem_norm = F.normalize(memory_bank, p=2, dim=1)   # (M, C, H, W)

        # 2. 為了實現 "Local Max"，我們使用 Unfold 將 Memory Bank 展開
        mem_unfold = F.unfold(mem_norm, kernel_size=self.kernel_size, padding=self.padding)
        mem_unfold = mem_unfold.view(m, c, self.kernel_size**2, h, w) # (M, C, K*K, H, W)

        # 3. 計算相似度
        similarity_maps = []
        
        for i in range(b):
            q_i = q_norm[i].unsqueeze(0).unsqueeze(2) # shape: (1, C, 1, H, W)
            
            # 計算 Cosine Similarity
            sim_i = (mem_unfold * q_i).sum(dim=1) 
            
            # 4. 局部最大化 : K*K 鄰域內找最大值
            pairwise_map, _ = torch.max(sim_i, dim=1)
            pairwise_map = torch.clamp(pairwise_map, min=0)  # (0~1)
            
            # 5. 群組聚合
            if self.use_product:
                # 連乘 (只要有一個正常樣本不像，分數就變 0)
                group_map = torch.prod(pairwise_map, dim=0, keepdim=True)
            else:
                # 平均 
                group_map = torch.mean(pairwise_map, dim=0, keepdim=True)
            
            similarity_maps.append(group_map)

        # Stack back to (B, 1, H, W)
        commonality_map = torch.stack(similarity_maps, dim=0)

        # 6. 反轉與特徵增強 (Inversion)
        anomaly_map = 1.0 - commonality_map
        
        return anomaly_map
    
class StaticMemoryBank:
    def __init__(self, device, max_size=5):
        self.max_size = max_size
        self.device = device
        self.bank = None
    
    def set_bank(self, features):
        """
        直接設定 Bank 的內容, 每個 Epoch 更新
        Args:
            features: 已經處理好形狀的 Tensor (N, C, H, W)
        """
        self.bank = features.detach()

    def get_bank(self):
        return self.bank


class Discriminator(torch.nn.Module):
    def __init__(self, in_planes, n_layers=2, hidden=None):
        super(Discriminator, self).__init__()

        _hidden = in_planes if hidden is None else hidden
        self.body = torch.nn.Sequential()
        for i in range(n_layers - 1):
            _in = in_planes if i == 0 else _hidden
            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            self.body.add_module('block%d' % (i + 1),
                                 torch.nn.Sequential(
                                     torch.nn.Linear(_in, _hidden),
                                     torch.nn.BatchNorm1d(_hidden),
                                     torch.nn.LeakyReLU(0.2)
                                 ))
        self.tail = torch.nn.Sequential(torch.nn.Linear(_hidden, 1, bias=False),
                                        torch.nn.Sigmoid())
        self.apply(init_weight)

    def forward(self, x):
        x = self.body(x)
        x = self.tail(x)
        return x


class Projection(torch.nn.Module):
    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0):
        super(Projection, self).__init__()

        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential()
        _in = None
        _out = None
        for i in range(n_layers):
            _in = in_planes if i == 0 else _out
            _out = out_planes
            self.layers.add_module(f"{i}fc", torch.nn.Linear(_in, _out))
            if i < n_layers - 1:
                if layer_type > 1:
                    self.layers.add_module(f"{i}relu", torch.nn.LeakyReLU(.2))
        self.apply(init_weight)

    def forward(self, x):

        x = self.layers(x)
        return x


class PatchMaker:
    def __init__(self, patchsize, top_k=0, stride=None):
        self.patchsize = patchsize
        self.stride = stride
        self.top_k = top_k

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1)
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (s + 2 * padding - 1 * (self.patchsize - 1) - 1) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        x = x[:, :, 0]
        x = torch.max(x, dim=1).values
        return x
