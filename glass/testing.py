import torch
import torchvision.models as models # 我們需要借用一個標準模型當骨幹
from glass.glass import GLASS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

randn_images = torch.randn(1, 3, 256, 256).to(device)

backbone = models.wide_resnet50_2(pretrained=False) 

gls = GLASS(device)

gls.load(
    backbone=backbone,
    layers_to_extract_from=["layer2", "layer3"], # 指定要抽特徵的層
    device=device,
    input_shape=(3, 256, 256),
    pretrain_embed_dimension=1536,
    target_embed_dimension=1536,
    patchsize=3,
    patchstride=1
)

patch_features, patch_shapes = gls._embed(randn_images, evaluation=True)

# 7. 查看結果
print("-" * 30)
print("Patch Features Shape:", patch_features.shape)
print("Patch Shapes:", patch_shapes)
print("-" * 30)