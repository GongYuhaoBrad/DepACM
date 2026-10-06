import torch
import torch.nn as nn
from DepthNet.base_model import BaseModel
from DepthNet.blocks import  FeatureFusionBlock_custom, Interpolate
def make_scratch(in_shape, out_shape, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape*2
        out_shape3 = out_shape*4
        if len(in_shape) >= 4:
            out_shape4 = out_shape*8

    scratch.layer1_rn = nn.Sequential(nn.Conv2d(in_shape[0], in_shape[0], kernel_size=3, stride=1, padding=1, bias=False, groups=in_shape[0]),
        nn.Conv2d(in_shape[0], out_shape1, kernel_size=1,  bias=False),)
    scratch.layer2_rn = nn.Sequential(nn.Conv2d(in_shape[1], in_shape[1], kernel_size=3, stride=1, padding=1, bias=False, groups=in_shape[1]),
        nn.Conv2d(in_shape[1], out_shape2, kernel_size=1, bias=False),)
    scratch.layer3_rn = nn.Sequential(nn.Conv2d(in_shape[2], in_shape[2], kernel_size=3, stride=1, padding=1, bias=False, groups=in_shape[2]),
        nn.Conv2d(in_shape[2], out_shape3, kernel_size=1,  bias=False),)
    if len(in_shape) >= 4:
        scratch.layer4_rn=nn.Sequential(nn.Conv2d(in_shape[3], in_shape[3], kernel_size=3, stride=1, padding=1, bias=False, groups=in_shape[3]),
        nn.Conv2d(in_shape[3], out_shape4, kernel_size=1, bias=False),)
    return scratch
def make_efficientnet_lite3(use_pretrained, exportable=False):
    efficientnet = torch.hub.load(
        "/home/customer4/.cache/torch/hub/rwightman_gen-efficientnet-pytorch_master/",
        "tf_efficientnet_lite0",
        pretrained=use_pretrained,
        exportable=exportable,
        source='local',
    )
    pretrained = nn.Module()

    pretrained.layer1 = nn.Sequential(
        efficientnet.conv_stem, efficientnet.bn1, efficientnet.act1, *efficientnet.blocks[0:2]
    )
    pretrained.layer2 = nn.Sequential(*efficientnet.blocks[2:3])
    pretrained.layer3 = nn.Sequential(*efficientnet.blocks[3:5])
    pretrained.layer4 = nn.Sequential(*efficientnet.blocks[5:9])

    return pretrained
def DSConvBS(in_shape, out_shape,expand=False):
    DSConvB = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape
    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape*2
        out_shape3 = out_shape*4
        if len(in_shape) >= 4:
            out_shape4 = out_shape*8
    DSConvB.layer1_rn = nn.Sequential(
        nn.Conv2d(in_shape[0], in_shape[0], kernel_size=3, stride=1, padding=1, bias=False, groups=in_shape[0]),
        nn.Conv2d(in_shape[0], out_shape1, kernel_size=1,  bias=False),)
    DSConvB.layer2_rn = nn.Sequential(
        nn.Conv2d(in_shape[1], in_shape[1], kernel_size=3, stride=1, padding=1, bias=False, groups=in_shape[1]),
        nn.Conv2d(in_shape[1], out_shape2, kernel_size=1, bias=False),)
    DSConvB.layer3_rn = nn.Sequential(
        nn.Conv2d(in_shape[2], in_shape[2], kernel_size=3, stride=1, padding=1, bias=False, groups=in_shape[2]),
        nn.Conv2d(in_shape[2], out_shape3, kernel_size=1,  bias=False),)
    if len(in_shape) >= 4:
        DSConvB.layer4_rn=nn.Sequential(
        nn.Conv2d(in_shape[3], in_shape[3], kernel_size=3, stride=1, padding=1, bias=False, groups=in_shape[3]),
        nn.Conv2d(in_shape[3], out_shape4, kernel_size=1, bias=False),)
    return DSConvB
class LUDEN(BaseModel):
    def __init__(self, path=None, features=64, backbone="efficientnet_lite3", non_negative=True, exportable=True, channels_last=False, align_corners=True,
        blocks={'expand': True}):

        print("Loading weights: ", path)

        super(LUDEN, self).__init__()
        use_pretrained = False if path else True
        print(backbone)
        self.channels_last = channels_last
        self.blocks = blocks
        self.backbone = backbone
        self.groups = 8
        features1=features
        features2=features
        features3=features
        features4=features
        self.expand = False
        if "expand" in self.blocks and self.blocks['expand'] == True:
            self.expand = True
            features1=features
            features2=features*2
            features3=features*4
            features4=features*8

        self.pretrained= make_efficientnet_lite3( use_pretrained, exportable=exportable)
        self.DSConvB =DSConvBS([24, 40, 112, 320], features,expand=self.expand)

        self.activation = nn.ReLU(True)    
        self.refinenet4 = FeatureFusionBlock_custom(features4, self.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.refinenet3 = FeatureFusionBlock_custom(features3, self.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.refinenet2 = FeatureFusionBlock_custom(features2, self.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.refinenet1 = FeatureFusionBlock_custom(features1, self.activation, deconv=False, bn=False, align_corners=align_corners)
        self.output_conv = nn.Sequential(
            nn.Conv2d(features, features//2, kernel_size=3, stride=1, padding=1, groups=self.groups),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(features//2, 32, kernel_size=3, stride=1, padding=1),
            self.activation,
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),#nn.Sigmoid(),#
            nn.Identity(),
        )
        
    

    def forward(self, x):
        if self.channels_last==True:
            print("self.channels_last = ", self.channels_last)
            x.contiguous(memory_format=torch.channels_last)

        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        layer_1_rn = self.DSConvB.layer1_rn(layer_1)
        layer_2_rn = self.DSConvB.layer2_rn(layer_2)
        layer_3_rn = self.DSConvB.layer3_rn(layer_3)
        layer_4_rn = self.DSConvB.layer4_rn(layer_4)

        path_4 = self.refinenet4(layer_4_rn)
        path_3 = self.refinenet3(path_4, layer_3_rn)
        path_2 = self.refinenet2(path_3, layer_2_rn)
        path_1 = self.refinenet1(path_2, layer_1_rn)
        
        out = self.output_conv(path_1)
        return out