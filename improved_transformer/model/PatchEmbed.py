from torch import nn



class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=nn.LayerNorm):
        super().__init__()
        self.img_size = [img_size, img_size]
        # patch大小
        self.patch_size = [patch_size, patch_size]
        # 每个方向上的patch数量
        patches_resolution = [self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        # patch总数
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        # 输入通道
        self.in_chans = in_chans
        # embed维度
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        # B 3 H,W  -> Ph,Pw = 224//4=56 -> B 96 Ph Pw -> B ph*pw 96
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

