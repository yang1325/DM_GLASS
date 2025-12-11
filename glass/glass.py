from glass.loss import FocalLoss, NTXentLoss
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from glass.model import Discriminator, Projection, PatchMaker, StaticMemoryBank, CoresetSampler, SpatialGCM, SampledGCM
from glass import backbones

import numpy as np
import pandas as pd
import torch.nn.functional as F

import logging
import os
import math
import torch
import tqdm
from glass import *
from glass import metrics, utils, common
import cv2
import glob
import shutil


LOGGER = logging.getLogger(__name__)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TBWrapper:
    def __init__(self, log_dir):
        self.g_iter = 0
        self.logger = SummaryWriter(log_dir=log_dir)

    def step(self):
        self.g_iter += 1


class GLASS(torch.nn.Module):
    # def __init__(self, device):
    #     super(GLASS, self).__init__()
    #     self.device = device

    def __init__(self, backbone,
            layers_to_extract_from,
            device,
            input_shape,
            pretrain_embed_dimension,
            target_embed_dimension, **kwargs):
        super(GLASS, self).__init__()
        self.device = device
        self.load(backbone,
            layers_to_extract_from,
            device,
            input_shape,
            pretrain_embed_dimension,
            target_embed_dimension, **kwargs)

    def load(
            self,
            backbone,
            layers_to_extract_from,
            device,
            input_shape,
            pretrain_embed_dimension,
            target_embed_dimension,
            patchsize=3,
            patchstride=1,
            meta_epochs=640,
            eval_epochs=1,
            dsc_layers=2,
            dsc_hidden=1024,
            dsc_margin=0.5,
            train_backbone=False,
            pre_proj=1,
            mining=1,
            noise=0.015,
            radius=0.75,
            p=0.5,
            lr=0.0001,
            svd=0,
            step=20,
            limit=392,
            memory_bank_type="sampled",
            sampling_ratio=0.1,
            **kwargs,
    ):
        self.contrastive_criterion = NTXentLoss()
        self.pretrain_embed_dimension = pretrain_embed_dimension
        if(type(backbone) == type("string")):
            backbone = backbones.load(backbone)
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        self.device = device

        self.forward_modules = torch.nn.ModuleDict({})
        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device, train_backbone
        )
        # Feature Extractor = feature_aggregator
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        # Feature Adaptor = preprocessing + "preadapt_aggregator
        preprocessing = common.Preprocessing(feature_dimensions, pretrain_embed_dimension)
        self.forward_modules["preprocessing"] = preprocessing
        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = common.Aggregator(target_dim=target_embed_dimension)
        preadapt_aggregator.to(self.device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.meta_epochs = meta_epochs
        self.lr = lr
        self.train_backbone = train_backbone
        if self.train_backbone:
            self.backbone_opt = torch.optim.AdamW(self.forward_modules["feature_aggregator"].backbone.parameters(), lr)

        # ================= Memory Bank Module =================
        self.memory_bank_type = memory_bank_type
        self.sampling_ratio = sampling_ratio

        self.memory_bank = StaticMemoryBank(max_size=5, device=self.device)
        if memory_bank_type == "spatial":
            self.gcm = SpatialGCM(kernel_size=3, use_product_aggregation=False)
        else:
            self.gcm = SampledGCM(use_product_aggregation=False)
        self.gcm.to(device=device)
        # =======================================================

        self.pre_proj = pre_proj
        if self.pre_proj > 0:
            self.pre_projection = Projection(self.target_embed_dimension, self.target_embed_dimension, pre_proj)
            self.pre_projection.to(self.device)
            self.proj_opt = torch.optim.Adam(self.pre_projection.parameters(), lr, weight_decay=1e-5)

        self.eval_epochs = eval_epochs
        self.dsc_layers = dsc_layers
        self.dsc_hidden = dsc_hidden
        self.discriminator = Discriminator(self.target_embed_dimension+1, n_layers=dsc_layers, hidden=dsc_hidden)
        self.discriminator.to(self.device)
        self.dsc_opt = torch.optim.AdamW(self.discriminator.parameters(), lr=lr * 2)
        self.dsc_margin = dsc_margin

        self.c = torch.tensor(0)
        self.c_ = torch.tensor(0)
        self.p = p
        self.radius = radius
        self.mining = mining
        self.noise = noise
        self.svd = svd
        self.step = step
        self.limit = limit
        self.distribution = 0
        self.focal_loss = FocalLoss()

        self.patch_maker = PatchMaker(patchsize, stride=patchstride)
        self.anomaly_segmentor = common.RescaleSegmentor(device=self.device, target_size=input_shape[-2:])
        self.model_dir = ""
        self.dataset_name = ""
        self.logger = None

    def set_model_dir(self, model_dir, dataset_name):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.model_dir, dataset_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.ckpt_dir, "tb")
        os.makedirs(self.tb_dir, exist_ok=True)
        self.logger = TBWrapper(self.tb_dir)

    def _embed(self, images, detach=True, provide_patch_shapes=False, evaluation=False):
        """Returns feature embeddings for images."""
        patch_features, patch_shapes = self._encode(images, detach=detach,
                                                    provide_patch_shapes=provide_patch_shapes,
                                                    evaluation=evaluation)
        patch_features = [x.reshape(-1, *x.shape[-3:]) for x in patch_features]
        patch_features = self.forward_modules["preprocessing"](patch_features)
        patch_features = self.forward_modules["preadapt_aggregator"](patch_features)

        return patch_features, patch_shapes
    
    def normalize_batch(self, imgs):
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(imgs.device)
        std  = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(imgs.device)
        return ((imgs+1)/2 - mean) / std

    def denormalize_batch(self, imgs):
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(imgs.device)
        std  = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(imgs.device)
        return (imgs * std + mean)*2 - 1

    def decode(self, images, detach=True, provide_patch_shapes=False, evaluation=False):
        images = images[:,:3,:,:]
        images = self.normalize_batch(images)
        return images
    
    def encode(self, images, detach=True, provide_patch_shapes=False, evaluation=False):
        images = self.denormalize_batch(images)
        b, c, h, w = images.shape
        zero_channel = images.new_zeros((b, 1, h, w))
        images = torch.cat([images, zero_channel], dim=1)
        return images
    
    def _encode(self, images, detach=True, provide_patch_shapes=False, evaluation=False):
        """Returns feature embeddings for images."""
        # print(images.dtype)
        if not evaluation and self.train_backbone:
            self.forward_modules["feature_aggregator"].train()
            features = self.forward_modules["feature_aggregator"](images, eval=evaluation)
        else:
            self.forward_modules["feature_aggregator"].eval()
            # print(images.shape)
            with torch.no_grad():
                features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]
        

        for i, feat in enumerate(features):
            # 把 ViT 圖片格式(B,L,C)調成一般格式(B,C,H,W)
            if len(feat.shape) == 3:
                B, L, C = feat.shape
                features[i] = feat.reshape(B, int(math.sqrt(L)), int(math.sqrt(L)), C).permute(0, 3, 1, 2)
        

        features = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features]
        patch_shapes = [x[1] for x in features]
        patch_features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(patch_features)):
            _features = patch_features[i]
            patch_dims = patch_shapes[i]

            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, 3, 4, 5, 1, 2)
            
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            # print(_features.permute(1,0,2,3).shape)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            # print(_features.shape, *perm_base_shape[:-2], ref_num_patches, patch_shapes)
            _features = _features.permute(0, 4, 5, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            patch_features[i] = _features
        patch_features = [torch.concat(patch_features, 2)]

        return patch_features, patch_shapes

        
    
    def _encode2(self, patch_features):
        patch_features = self.forward_modules["preprocessing"](patch_features)
        return patch_features


    def init_memory_bank(self, support_images):
        self.forward_modules["feature_aggregator"].eval()
        self.forward_modules["preprocessing"].eval()
        self.forward_modules["preadapt_aggregator"].eval()
        if self.pre_proj > 0:
            self.pre_projection.eval()

        with torch.no_grad():
            features, patch_shapes = self._embed(support_images, evaluation = True)
            if self.pre_proj > 0:
                features = self.pre_projection(features)
                if isinstance(features, (list, tuple)):
                    features = features[0]

            self.spatial_h = patch_shapes[0][0]
            self.spatial_w = patch_shapes[0][1]

            n = support_images.size(0)
            c = features.shape[-1]
        
            features_spatial = features.reshape(n, self.spatial_h * self.spatial_w, c).permute(0, 2, 1).reshape(n, c, self.spatial_h, self.spatial_w)
            self.memory_bank.set_bank(features_spatial)

        # print(f"Bank Initialized. Dynamic Spatial Shape: ({self.spatial_h}, {self.spatial_w})")
        if self.train_backbone:
            self.forward_modules["feature_aggregator"].train()
        self.forward_modules["preprocessing"].train()
        self.forward_modules["preadapt_aggregator"].train()
        if self.pre_proj > 0:
            self.pre_projection.train()

    # def enhance_features(self, flat_features, batch_size):
    #     """
    #     將原始特徵加入 Memory Bank 的比對資訊。
    #     Args:
    #         flat_features: _embed 輸出的特徵 (Batch*1024, 1536)
    #         batch_size: 當前的 Batch Size
    #     Returns:
    #         enhanced_features: (Batch*1024, 1537)
    #     """
    #     # 1. 準備維度參數
    #     c = self.target_embed_dimension
    #     h, w = self.spatial_h, self.spatial_w
        
    #     # 2. Reshape: Flatten -> Spatial
    #     # (B*1024, 1536) -> (B, 1536, 32, 32)
    #     features_spatial = flat_features.reshape(batch_size, h*w, c).permute(0, 2, 1).reshape(batch_size, c, h, w)
        
    #     # 3. GCM 比對 (得到 Anomaly Map)
    #     print(self.memory_bank.get_bank().shape)

    #     print(features_spatial.shape)
    #     anomaly_map = self.gcm(features_spatial, self.memory_bank.get_bank())
        
    #     # 4. Flatten Anomaly Map
    #     # (B, 1, 32, 32) -> (B, 32*32, 1) -> (B*1024, 1)
    #     anomaly_map_flat = anomaly_map.reshape(batch_size, 1, h*w).permute(0, 2, 1).reshape(-1, 1)
        
    #     # 5. 特徵串接 (Concatenate)
    #     # (B*1024, 1536) cat (B*1024, 1) -> (B*1024, 1537)
    #     enhanced_features = torch.cat([flat_features, anomaly_map_flat], dim=1)
        
    #     return enhanced_features

    def enhance_features(self, flat_features, batch_size):
        c = self.target_embed_dimension
        h, w = self.spatial_h, self.spatial_w

        # (B*1024,1536) -> (B,1536,32,32)
        features_spatial = (
            flat_features.reshape(batch_size, h*w, c)
            .permute(0, 2, 1)
            .reshape(batch_size, c, h, w)
        )

        ##############################################
        # 修正：memory bank 必須轉為 [M, C]
        ##############################################
        mem = self.memory_bank.get_bank()

        if mem.dim() == 4:  
            # [1, C, H, W] → [H*W, C]
            mem = mem.permute(0, 2, 3, 1).reshape(-1, c)  # [1296, 1536]

        ##############################################
        # 呼叫 GCM 正常運作
        ##############################################
        anomaly_map = self.gcm(features_spatial, mem)   # mem: [M, C]

        ##############################################
        # flatten
        ##############################################
        anomaly_map_flat = (
            anomaly_map.reshape(batch_size, 1, h*w)
            .permute(0, 2, 1)
            .reshape(-1, 1)
        )

        enhanced_features = torch.cat([flat_features, anomaly_map_flat], dim=1)
        return enhanced_features


    def trainer(self, training_data, val_data, name):
        state_dict = {}
        
        # NEW: Re-add the missing nested function definition
        def update_state_dict():
            state_dict["discriminator"] = OrderedDict({
                k: v.detach().cpu()
                for k, v in self.discriminator.state_dict().items()})
            if self.pre_proj > 0:
                state_dict["pre_projection"] = OrderedDict({
                    k: v.detach().cpu()
                    for k, v in self.pre_projection.state_dict().items()})
        # --- END OF FIX ---

        # 1. MODIFIED: Define paths and check for resume
        ckpt_path_best_list = glob.glob(self.ckpt_dir + '/ckpt_best*') # List of 'best' checkpoints
        ckpt_path_save = os.path.join(self.ckpt_dir, "ckpt.pth") # The 'latest' checkpoint for resuming
        
        start_epoch = 0 # NEW: Default start epoch
        best_record = None # NEW: Default best record
        ckpt_path_best = None # NEW: Variable to track the current best checkpoint path

        # Check if we are resuming (i.e., 'best' checkpoints exist)
        if len(ckpt_path_best_list) > 0:
            LOGGER.info(f"Found existing checkpoints. Attempting to resume training...")
            
            # We must load the *latest* state to continue training, which is 'ckpt.pth'
            if os.path.exists(ckpt_path_save):
                LOGGER.info(f"Loading latest state from {ckpt_path_save}")
                # Load the state dict first
                loaded_state = torch.load(ckpt_path_save, map_location=self.device)
                if 'discriminator' in loaded_state:
                    self.discriminator.load_state_dict(loaded_state['discriminator'])
                if "pre_projection" in loaded_state:
                    self.pre_projection.load_state_dict(loaded_state["pre_projection"])
                
                # IMPORTANT: Update the 'state_dict' variable so 'update_state_dict' can work on it
                state_dict = loaded_state 
            else:
                # Fallback in case 'ckpt.pth' is missing but 'best' exists
                LOGGER.warning(f"Could not find 'latest' {ckpt_path_save}, loading 'best' as starting point.")
                best_ckpt_to_load = sorted(ckpt_path_best_list, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
                loaded_state = torch.load(best_ckpt_to_load, map_location=self.device)
                if 'discriminator' in loaded_state:
                    self.discriminator.load_state_dict(loaded_state['discriminator'])
                if "pre_projection" in loaded_state:
                    self.pre_projection.load_state_dict(loaded_state["pre_projection"])
                
                # IMPORTANT: Update the 'state_dict' variable
                state_dict = loaded_state

            # Find the latest epoch from the 'best' file names to set the start epoch
            epochs = [int(p.split('_')[-1].split('.')[0]) for p in ckpt_path_best_list]
            start_epoch = max(epochs) + 1
            LOGGER.info(f"Resuming training from epoch {start_epoch}")
            
            # We must also restore the 'best_record' or the save logic will fail
            # We do this by re-running evaluation on the *best* model found
            ckpt_path_best = sorted(ckpt_path_best_list, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
            last_best_epoch = int(ckpt_path_best.split('_')[-1].split('.')[0])
            
            LOGGER.info(f"Restoring best_record from {ckpt_path_best} (Epoch {last_best_epoch})...")
            
            # Temporarily load the *best* model state to get its score
            best_state_dict = torch.load(ckpt_path_best, map_location=self.device)
            temp_discriminator_state = self.discriminator.state_dict()
            temp_projection_state = self.pre_projection.state_dict() if self.pre_proj > 0 else None
            
            self.discriminator.load_state_dict(best_state_dict['discriminator'])
            if self.pre_proj > 0 and "pre_projection" in best_state_dict:
                self.pre_projection.load_state_dict(best_state_dict["pre_projection"])

            # Run evaluation to get the scores
            images, scores, segmentations, labels_gt, masks_gt = self.predict(val_data)
            i_auroc, i_ap, p_auroc, p_ap, p_pro = self._evaluate(images, scores, segmentations,
                                                                 labels_gt, masks_gt, name, path='resume_eval')
            
            best_record = [i_auroc, i_ap, p_auroc, p_ap, p_pro, last_best_epoch]
            LOGGER.info(f"Restored best_record: {best_record}")

            # Restore the *latest* model state to continue training
            LOGGER.info(f"Reloading latest state from {ckpt_path_save} to continue...")
            self.discriminator.load_state_dict(temp_discriminator_state)
            if self.pre_proj > 0 and temp_projection_state is not None:
                self.pre_projection.load_state_dict(temp_projection_state)

        # --- END OF MODIFIED RESUME BLOCK ---

        self.distribution = training_data.dataset.distribution
        xlsx_path = './datasets/excel/' + name.split('_')[0] + '_distribution.xlsx'
        try:
            if self.distribution == 1:  # rejudge by image-level spectrogram analysis
                self.distribution = 1
                self.svd = 1
            elif self.distribution == 2:  # manifold
                self.distribution = 0
                self.svd = 0
            elif self.distribution == 3:  # hypersphere
                self.distribution = 0
                self.svd = 1
            elif self.distribution == 4:  # opposite choose by file
                self.distribution = 0
                df = pd.read_excel(xlsx_path)
                self.svd = 1 - df.loc[df['Class'] == name, 'Distribution'].values[0]
            else:  # choose by file
                self.distribution = 0
                df = pd.read_excel(xlsx_path)
                self.svd = df.loc[df['Class'] == name, 'Distribution'].values[0]
        except:
            self.distribution = 1
            self.svd = 1

        # judge by image-level spectrogram analysis
        if self.distribution == 1:
            self.forward_modules.eval()
            with torch.no_grad():
                for i, data in enumerate(training_data):
                    img = data["image"]
                    img = img.to(torch.float).to(self.device)
                    batch_mean = torch.mean(img, dim=0)
                    if i == 0:
                        self.c = batch_mean
                    else:
                        self.c += batch_mean
                self.c /= len(training_data)

            avg_img = utils.torch_format_2_numpy_img(self.c.detach().cpu().numpy())
            self.svd = utils.distribution_judge(avg_img, name)
            os.makedirs(f'./results/judge/avg/{self.svd}', exist_ok=True)
            cv2.imwrite(f'./results/judge/avg/{self.svd}/{name}.png', avg_img)
            return self.svd

        # 2. MODIFIED: Use start_epoch in tqdm
        pbar = tqdm.tqdm(range(start_epoch, self.meta_epochs), unit='epoch')
        pbar_str1 = ""
        # best_record = None # <-- This is now set above
        
        for i_epoch in pbar:
            self.forward_modules.eval()

            support_images_list = []
            with torch.no_grad():  # compute center
                for i, data in enumerate(training_data):
                    img = data["image"]
                    img = img.to(torch.float).to(self.device)

                    support_images_list.append(img)

                    if self.pre_proj > 0:
                        outputs = self.pre_projection(self._embed(img, evaluation=False)[0])
                        outputs = outputs[0] if len(outputs) == 2 else outputs
                    else:
                        outputs = self._embed(img, evaluation=False)[0]
                    outputs = outputs[0] if len(outputs) == 2 else outputs
                    outputs = outputs.reshape(img.shape[0], -1, outputs.shape[-1])

                    batch_mean = torch.mean(outputs, dim=0)
                    if i == 0:
                        self.c = batch_mean
                    else:
                        self.c += batch_mean
                self.c /= len(training_data)

            # --- 初始化 Memory Bank ---
            # 把不同批次的image合併
            support_images = torch.cat(support_images_list, dim=0)
        
            self.init_memory_bank(support_images)
            # ------------------------------
                
            pbar_str, pt, pf = self._train_discriminator(training_data, i_epoch, pbar, pbar_str1)
            
            # This call should now work
            update_state_dict() 

            if (i_epoch + 1) % self.eval_epochs == 0:
                images, scores, segmentations, labels_gt, masks_gt = self.predict(val_data)
                image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro = self._evaluate(images, scores, segmentations,
                                                                                         labels_gt, masks_gt, name)

                self.logger.logger.add_scalar("i-auroc", image_auroc, i_epoch)
                self.logger.logger.add_scalar("i-ap", image_ap, i_epoch)
                self.logger.logger.add_scalar("p-auroc", pixel_auroc, i_epoch)
                self.logger.logger.add_scalar("p-ap", pixel_ap, i_epoch)
                self.logger.logger.add_scalar("p-pro", pixel_pro, i_epoch)

                eval_path = './results/eval/' + name + '/'
                train_path = './results/training/' + name + '/'
                
                # 3. MODIFIED: Fix checkpoint saving/deleting logic
                if best_record is None or (image_auroc + pixel_auroc > best_record[0] + best_record[2]):
                    best_record = [image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, i_epoch]
                    
                    # Remove the *previous* best checkpoint, if it exists
                    if ckpt_path_best is not None and os.path.exists(ckpt_path_best):
                        os.remove(ckpt_path_best)
                        
                    # Define and save the *new* best checkpoint
                    ckpt_path_best = os.path.join(self.ckpt_dir, "ckpt_best_{}.pth".format(i_epoch))
                    torch.save(state_dict, ckpt_path_best)
                    
                    # Copy visualization files
                    shutil.rmtree(eval_path, ignore_errors=True)
                    shutil.copytree(train_path, eval_path)

                pbar_str1 = f" IAUC:{round(image_auroc * 100, 2)}({round(best_record[0] * 100, 2)})" \
                            f" IAP:{round(image_ap * 100, 2)}({round(best_record[1] * 100, 2)})" \
                            f" PAUC:{round(pixel_auroc * 100, 2)}({round(best_record[2] * 100, 2)})" \
                            f" PAP:{round(pixel_ap * 100, 2)}({round(best_record[3] * 100, 2)})" \
                            f" PRO:{round(pixel_pro * 100, 2)}({round(best_record[4] * 100, 2)})" \
                            f" E:{i_epoch}({best_record[-1]})"
                pbar_str += pbar_str1
                pbar.set_description_str(pbar_str)

            # Save the 'latest' checkpoint every epoch
            torch.save(state_dict, ckpt_path_save)
        
        self.logger.logger.close()
        return best_record
    def train_one_batch(self, batch):
        

        self.dsc_opt.zero_grad()
        if self.pre_proj > 0:
            self.proj_opt.zero_grad()

        view1 = batch["view1"].to(torch.float).to(self.device)
        view2 = batch["view2"].to(torch.float).to(self.device)
        
        view1_feat, _ = self._embed(view1, evaluation=False)
        view2_feat, _ = self._embed(view2, evaluation=False)

        if self.pre_proj>0:
            view1_feat = self.pre_projection(view1_feat)
            view2_feat = self.pre_projection(view2_feat)
        
        loss_clr = self.contrastive_criterion(view1_feat, view2_feat)
        avg_sim = torch.nn.functional.cosine_similarity(view1_feat, view2_feat).mean()

        aug = batch["aug"]
        aug = aug.to(torch.float).to(self.device)
        aug2 = batch["diff"]
        aug2 = aug2.to(torch.float).to(self.device)
        img = batch["image"]
        img = img.to(torch.float).to(self.device)
        if self.pre_proj > 0:
            fake_feats = self.pre_projection(self._embed(aug, evaluation=False)[0])
            fake_feats = fake_feats[0] if len(fake_feats) == 2 else fake_feats
            fake_feats2 = self.pre_projection(self._embed(aug2, evaluation=False)[0])
            fake_feats2 = fake_feats2[0] if len(fake_feats2) == 2 else fake_feats2
            true_feats = self.pre_projection(self._embed(img, evaluation=False)[0])
            true_feats = true_feats[0] if len(true_feats) == 2 else true_feats
        else:
            fake_feats = self._embed(aug, evaluation=False)[0]
            fake_feats.requires_grad = True
            fake_feats2 = self._embed(aug2, evaluation=False)[0]
            fake_feats2.requires_grad = True
            true_feats = self._embed(img, evaluation=False)[0]
            true_feats.requires_grad = True

        mask_s_gt = batch["mask_s"].reshape(-1, 1).to(self.device)
        noise = torch.normal(0, self.noise, true_feats.shape).to(self.device)
        gaus_feats = true_feats + noise

        center = self.c.repeat(img.shape[0], 1, 1)
        center = center.reshape(-1, center.shape[-1])
        true_points = torch.concat([fake_feats[mask_s_gt[:, 0] == 0],fake_feats2[mask_s_gt[:, 0] == 0], true_feats], dim=0)
        c_t_points = torch.concat([center[mask_s_gt[:, 0] == 0], center[mask_s_gt[:, 0] == 0], center], dim=0)
        dist_t = torch.norm(true_points - c_t_points, dim=1)
        r_t = torch.tensor([torch.quantile(dist_t, q=self.radius)]).to(self.device)

        for step in range(self.step + 1):
            # 透過 Memory Bank 增強特徵
            bs = img.shape[0]
            enhanced_true = self.enhance_features(true_feats, bs)
            enhanced_guas = self.enhance_features(gaus_feats, bs)

            scores = self.discriminator(torch.cat([enhanced_true, enhanced_guas]))

            true_scores = scores[:len(true_feats)]
            gaus_scores = scores[len(true_feats):]
            true_loss = torch.nn.BCELoss()(true_scores, torch.zeros_like(true_scores))
            gaus_loss = torch.nn.BCELoss()(gaus_scores, torch.ones_like(gaus_scores))
            bce_loss = true_loss + gaus_loss

            if step == self.step:
                break
            elif self.mining == 0:
                dist_g = torch.norm(gaus_feats - center, dim=1)
                r_g = torch.tensor([torch.quantile(dist_g, q=self.radius)]).to(self.device)
                break

            grad = torch.autograd.grad(gaus_loss, [gaus_feats])[0]
            grad_norm = torch.norm(grad, dim=1)
            grad_norm = grad_norm.view(-1, 1)
            grad_normalized = grad / (grad_norm + 1e-10)

            with torch.no_grad():
                gaus_feats.add_(0.001 * grad_normalized)

            if (step + 1) % 5 == 0:
                dist_g = torch.norm(gaus_feats - center, dim=1)
                r_g = torch.tensor([torch.quantile(dist_g, q=self.radius)]).to(self.device)
                proj_feats = center if self.svd == 1 else true_feats
                r = r_t if self.svd == 1 else 0.5

                h = gaus_feats - proj_feats
                h_norm = dist_g if self.svd == 1 else torch.norm(h, dim=1)
                alpha = torch.clamp(h_norm, r, 2 * r)
                proj = (alpha / (h_norm + 1e-10)).view(-1, 1)
                h = proj * h
                gaus_feats = proj_feats + h

        fake_points = fake_feats[mask_s_gt[:, 0] == 1]
        fake_points2 = fake_feats2[mask_s_gt[:, 0] == 1]
        true_points = true_feats[mask_s_gt[:, 0] == 1]
        c_f_points = center[mask_s_gt[:, 0] == 1]
        dist_f = torch.norm(fake_points - c_f_points, dim=1)
        dist_f2 = torch.norm(fake_points2 - c_f_points, dim=1)
        r_f = torch.tensor([torch.quantile(dist_f, q=self.radius)]).to(self.device)
        proj_feats = c_f_points if self.svd == 1 else true_points
        r = r_t if self.svd == 1 else 1

        if self.svd == 1:
            h = fake_points - proj_feats
            h2 = fake_points2 - proj_feats
            h_norm = dist_f if self.svd == 1 else torch.norm(h, dim=1)
            h_norm2 = dist_f2 if self.svd == 1 else torch.norm(h2, dim=1)
            alpha = torch.clamp(h_norm, 2 * r, 4 * r)
            alpha2 = torch.clamp(h_norm2, 2 * r, 4 * r)
            proj = (alpha / (h_norm + 1e-10)).view(-1, 1)
            proj2 = (alpha2 / (h_norm2 + 1e-10)).view(-1, 1)
            h = proj * h
            h2 = proj2 * h2
            fake_points = proj_feats + h
            fake_points2 = proj_feats + h2
            fake_feats[mask_s_gt[:, 0] == 1] = fake_points
            fake_feats2[mask_s_gt[:, 0] == 1] = fake_points2

        # 透過 Memory Bank 增強特徵
        enhanced_fake = self.enhance_features(fake_feats, bs)
        enhanced_fake2 = self.enhance_features(fake_feats2, bs)
        fake_scores = self.discriminator(enhanced_fake)
        fake_scores2 = self.discriminator(enhanced_fake2)

        if self.p > 0:
            fake_dist = (fake_scores - mask_s_gt) ** 2
            fake_dist2 = (fake_scores2 - mask_s_gt) ** 2
            d_hard = torch.quantile(fake_dist, q=self.p)
            d_hard2 = torch.quantile(fake_dist2, q=self.p)
            fake_scores_ = fake_scores[fake_dist >= d_hard].unsqueeze(1)
            fake_scores_2 = fake_scores2[fake_dist2 >= d_hard2].unsqueeze(1)
            mask_ = mask_s_gt[fake_dist >= d_hard].unsqueeze(1)
            mask_2 = mask_s_gt[fake_dist2 >= d_hard2].unsqueeze(1)
        else:
            fake_scores_ = fake_scores
            fake_scores_2 = fake_scores2
            mask_ = mask_s_gt
            mask_2 = mask_s_gt
        output = torch.cat([1 - fake_scores_, fake_scores_], dim=1)
        output2 = torch.cat([1 - fake_scores_2, fake_scores_2], dim=1)
        # print(output.shape,output2.shape,mask_2.shape, mask_.shape)
        focal_loss = self.focal_loss(output, mask_)
        focal_loss2 = self.focal_loss(output2, mask_2)

        loss = bce_loss + focal_loss + focal_loss2
        # loss.backward()
        # if self.pre_proj > 0:
        #     self.proj_opt.step()
        # if self.train_backbone:
        #     self.backbone_opt.step()
        # self.dsc_opt.step()


        return loss

    def _train_discriminator(self, input_data, cur_epoch, pbar, pbar_str1):
        self.forward_modules.eval()
        if self.pre_proj > 0:
            self.pre_projection.train()
        self.discriminator.train()

        all_loss, all_p_true, all_p_fake, all_r_t, all_r_g, all_r_f = [], [], [], [], [], []
        sample_num = 0
        for i_iter, data_item in enumerate(input_data):
            self.dsc_opt.zero_grad()
            if self.pre_proj > 0:
                self.proj_opt.zero_grad()

            aug = data_item["aug"]
            aug = aug.to(torch.float).to(self.device)
            img = data_item["image"]
            img = img.to(torch.float).to(self.device)
            if self.pre_proj > 0:
                fake_feats = self.pre_projection(self._embed(aug, evaluation=False)[0])
                fake_feats = fake_feats[0] if len(fake_feats) == 2 else fake_feats
                true_feats = self.pre_projection(self._embed(img, evaluation=False)[0])
                true_feats = true_feats[0] if len(true_feats) == 2 else true_feats
            else:
                fake_feats = self._embed(aug, evaluation=False)[0]
                fake_feats.requires_grad = True
                true_feats = self._embed(img, evaluation=False)[0]
                true_feats.requires_grad = True

            mask_s_gt = data_item["mask_s"].reshape(-1, 1).to(self.device)
            noise = torch.normal(0, self.noise, true_feats.shape).to(self.device)
            gaus_feats = true_feats + noise

            center = self.c.repeat(img.shape[0], 1, 1)
            center = center.reshape(-1, center.shape[-1])
            true_points = torch.concat([fake_feats[mask_s_gt[:, 0] == 0], true_feats], dim=0)
            c_t_points = torch.concat([center[mask_s_gt[:, 0] == 0], center], dim=0)
            dist_t = torch.norm(true_points - c_t_points, dim=1)
            r_t = torch.tensor([torch.quantile(dist_t, q=self.radius)]).to(self.device)

            for step in range(self.step + 1):
                # 透過 Memory Bank 增強特徵
                bs = img.shape[0]
                enhanced_true = self.enhance_features(true_feats, bs)
                enhanced_guas = self.enhance_features(gaus_feats, bs)

                scores = self.discriminator(torch.cat([enhanced_true, enhanced_guas]))

                true_scores = scores[:len(true_feats)]
                gaus_scores = scores[len(true_feats):]
                true_loss = torch.nn.BCELoss()(true_scores, torch.zeros_like(true_scores))
                gaus_loss = torch.nn.BCELoss()(gaus_scores, torch.ones_like(gaus_scores))
                bce_loss = true_loss + gaus_loss

                if step == self.step:
                    break
                elif self.mining == 0:
                    dist_g = torch.norm(gaus_feats - center, dim=1)
                    r_g = torch.tensor([torch.quantile(dist_g, q=self.radius)]).to(self.device)
                    break

                grad = torch.autograd.grad(gaus_loss, [gaus_feats])[0]
                grad_norm = torch.norm(grad, dim=1)
                grad_norm = grad_norm.view(-1, 1)
                grad_normalized = grad / (grad_norm + 1e-10)

                with torch.no_grad():
                    gaus_feats.add_(0.001 * grad_normalized)

                if (step + 1) % 5 == 0:
                    dist_g = torch.norm(gaus_feats - center, dim=1)
                    r_g = torch.tensor([torch.quantile(dist_g, q=self.radius)]).to(self.device)
                    proj_feats = center if self.svd == 1 else true_feats
                    r = r_t if self.svd == 1 else 0.5

                    h = gaus_feats - proj_feats
                    h_norm = dist_g if self.svd == 1 else torch.norm(h, dim=1)
                    alpha = torch.clamp(h_norm, r, 2 * r)
                    proj = (alpha / (h_norm + 1e-10)).view(-1, 1)
                    h = proj * h
                    gaus_feats = proj_feats + h

            fake_points = fake_feats[mask_s_gt[:, 0] == 1]
            true_points = true_feats[mask_s_gt[:, 0] == 1]
            c_f_points = center[mask_s_gt[:, 0] == 1]
            dist_f = torch.norm(fake_points - c_f_points, dim=1)
            r_f = torch.tensor([torch.quantile(dist_f, q=self.radius)]).to(self.device)
            proj_feats = c_f_points if self.svd == 1 else true_points
            r = r_t if self.svd == 1 else 1

            if self.svd == 1:
                h = fake_points - proj_feats
                h_norm = dist_f if self.svd == 1 else torch.norm(h, dim=1)
                alpha = torch.clamp(h_norm, 2 * r, 4 * r)
                proj = (alpha / (h_norm + 1e-10)).view(-1, 1)
                h = proj * h
                fake_points = proj_feats + h
                fake_feats[mask_s_gt[:, 0] == 1] = fake_points

             # 透過 Memory Bank 增強特徵
            enhanced_fake = self.enhance_features(fake_feats, bs)
            fake_scores = self.discriminator(enhanced_fake)

            if self.p > 0:
                fake_dist = (fake_scores - mask_s_gt) ** 2
                d_hard = torch.quantile(fake_dist, q=self.p)
                fake_scores_ = fake_scores[fake_dist >= d_hard].unsqueeze(1)
                mask_ = mask_s_gt[fake_dist >= d_hard].unsqueeze(1)
            else:
                fake_scores_ = fake_scores
                mask_ = mask_s_gt
            output = torch.cat([1 - fake_scores_, fake_scores_], dim=1)
            focal_loss = self.focal_loss(output, mask_)

            loss = bce_loss + focal_loss
            loss.backward()
            if self.pre_proj > 0:
                self.proj_opt.step()
            if self.train_backbone:
                self.backbone_opt.step()
            self.dsc_opt.step()

            pix_true = torch.concat([fake_scores.detach() * (1 - mask_s_gt), true_scores.detach()])
            pix_fake = torch.concat([fake_scores.detach() * mask_s_gt, gaus_scores.detach()])
            p_true = ((pix_true < self.dsc_margin).sum() - (pix_true == 0).sum()) / ((mask_s_gt == 0).sum() + true_scores.shape[0])
            p_fake = (pix_fake >= self.dsc_margin).sum() / ((mask_s_gt == 1).sum() + gaus_scores.shape[0])

            self.logger.logger.add_scalar(f"p_true", p_true, self.logger.g_iter)
            self.logger.logger.add_scalar(f"p_fake", p_fake, self.logger.g_iter)
            self.logger.logger.add_scalar(f"r_t", r_t, self.logger.g_iter)
            self.logger.logger.add_scalar(f"r_g", r_g, self.logger.g_iter)
            self.logger.logger.add_scalar(f"r_f", r_f, self.logger.g_iter)
            self.logger.logger.add_scalar("loss", loss, self.logger.g_iter)
            self.logger.step()

            all_loss.append(loss.detach().cpu().item())
            all_p_true.append(p_true.cpu().item())
            all_p_fake.append(p_fake.cpu().item())
            all_r_t.append(r_t.cpu().item())
            all_r_g.append(r_g.cpu().item())
            all_r_f.append(r_f.cpu().item())

            all_loss_ = np.mean(all_loss)
            all_p_true_ = np.mean(all_p_true)
            all_p_fake_ = np.mean(all_p_fake)
            all_r_t_ = np.mean(all_r_t)
            all_r_g_ = np.mean(all_r_g)
            all_r_f_ = np.mean(all_r_f)
            sample_num = sample_num + img.shape[0]

            pbar_str = f"epoch:{cur_epoch} loss:{all_loss_:.2e}"
            pbar_str += f" pt:{all_p_true_ * 100:.2f}"
            pbar_str += f" pf:{all_p_fake_ * 100:.2f}"
            pbar_str += f" rt:{all_r_t_:.2f}"
            pbar_str += f" rg:{all_r_g_:.2f}"
            pbar_str += f" rf:{all_r_f_:.2f}"
            pbar_str += f" svd:{self.svd}"
            pbar_str += f" sample:{sample_num}"
            pbar_str2 = pbar_str
            pbar_str += pbar_str1
            pbar.set_description_str(pbar_str)

            if sample_num > self.limit:
                break

        return pbar_str2, all_p_true_, all_p_fake_

    def tester(self, test_data, name):
        ckpt_path = glob.glob(self.ckpt_dir + '/ckpt_best*')
        if len(ckpt_path) != 0:
            state_dict = torch.load(ckpt_path[0], map_location=self.device)
            if 'discriminator' in state_dict:
                self.discriminator.load_state_dict(state_dict['discriminator'])
                if "pre_projection" in state_dict:
                    self.pre_projection.load_state_dict(state_dict["pre_projection"])
            else:
                self.load_state_dict(state_dict, strict=False)

            images, scores, segmentations, labels_gt, masks_gt = self.predict(test_data)
            image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro = self._evaluate(images, scores, segmentations,
                                                                                     labels_gt, masks_gt, name, path='eval')
            epoch = int(ckpt_path[0].split('_')[-1].split('.')[0])
        else:
            images, scores, segmentations, labels_gt, masks_gt = self.predict(test_data)
            image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro = self._evaluate(images, scores, segmentations,
                                                                                     labels_gt, masks_gt, name, path='eval')
            epoch = 0

        return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, epoch

    def _evaluate(self, images, scores, segmentations, labels_gt, masks_gt, name, path='training'):
        scores = np.squeeze(np.array(scores))
        image_scores = metrics.compute_imagewise_retrieval_metrics(scores, labels_gt, path)
        image_auroc = image_scores["auroc"]
        image_ap = image_scores["ap"]

        if len(masks_gt) > 0:
            segmentations = np.array(segmentations)
            pixel_scores = metrics.compute_pixelwise_retrieval_metrics(segmentations, masks_gt, path)
            pixel_auroc = pixel_scores["auroc"]
            pixel_ap = pixel_scores["ap"]
            if path == 'eval':
                try:
                    pixel_pro = metrics.compute_pro(np.squeeze(np.array(masks_gt)), segmentations)
                except:
                    pixel_pro = 0.
            else:
                pixel_pro = 0.
        else:
            pixel_auroc = -1.
            pixel_ap = -1.
            pixel_pro = -1.
            return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro

        defects = np.array(images)
        targets = np.array(masks_gt)
        for i in range(len(defects)):
            defect = utils.torch_format_2_numpy_img(defects[i])
            target = utils.torch_format_2_numpy_img(targets[i])

            mask = cv2.cvtColor(cv2.resize(segmentations[i], (defect.shape[1], defect.shape[0])),
                                cv2.COLOR_GRAY2BGR)
            mask = (mask * 255).astype('uint8')
            mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

            img_up = np.hstack([defect, target, mask])
            img_up = cv2.resize(img_up, (256 * 3, 256))
            full_path = './results/' + path + '/' + name + '/'
            utils.del_remake_dir(full_path, del_flag=False)
            cv2.imwrite(full_path + str(i + 1).zfill(3) + '.png', img_up)

        return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro

    def predict(self, test_dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        self.forward_modules.eval()

        img_paths = []
        images = []
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []

        with tqdm.tqdm(test_dataloader, desc="Inferring...", leave=False, unit='batch') as data_iterator:
            for data in data_iterator:
                if isinstance(data, dict):
                    labels_gt.extend(data["is_anomaly"].numpy().tolist())
                    if data.get("mask_gt", None) is not None:
                        masks_gt.extend(data["mask_gt"].numpy().tolist())
                    image = data["image"]
                    images.extend(image.numpy().tolist())
                    img_paths.extend(data["image_path"])
                _scores, _masks = self._predict(image)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)

        return images, scores, masks, labels_gt, masks_gt

    def _predict(self, img):
        """Infer score and mask for a batch of images."""
        img = img.to(torch.float).to(self.device)
        self.forward_modules.eval()

        if self.pre_proj > 0:
            self.pre_projection.eval()
        self.discriminator.eval()

        with torch.no_grad():

            patch_features, patch_shapes = self._embed(img, provide_patch_shapes=True, evaluation=True)
            if self.pre_proj > 0:
                patch_features = self.pre_projection(patch_features)
                patch_features = patch_features[0] if len(patch_features) == 2 else patch_features

            # === 特徵增強 ===
            bs = img.shape[0]
            enhanced_patch = self.enhance_features(patch_features, bs)
            patch_scores = image_scores = self.discriminator(enhanced_patch)

            patch_scores = self.patch_maker.unpatch_scores(patch_scores, batchsize=img.shape[0])
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(img.shape[0], scales[0], scales[1])
            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

            image_scores = self.patch_maker.unpatch_scores(image_scores, batchsize=img.shape[0])
            image_scores = self.patch_maker.score(image_scores)
            if isinstance(image_scores, torch.Tensor):
                image_scores = image_scores.cpu().numpy()

        return list(image_scores), list(masks)
