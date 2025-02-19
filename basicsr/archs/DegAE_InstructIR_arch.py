import torch.nn as nn

from basicsr.archs.Restormer_Backbone_arch import Restormer_Backbone
from basicsr.archs.Instructir_head_arch import InstructIR_head

from basicsr.utils.registry import ARCH_REGISTRY
@ARCH_REGISTRY.register()
class DegAE_InstructIR(nn.Module):
    def __init__(self, inp_channels=3, out_channels=64, dim=48, num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
                 heads=[1, 2, 4, 8], ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias',
                 global_residual=False, dual_pixel_task=False,
                 text_dim=512,
                 decoder_blk_nums=[2,2,2,2],
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = Restormer_Backbone(inp_channels, out_channels, dim, num_blocks, num_refinement_blocks, heads, ffn_expansion_factor, bias, LayerNorm_type, global_residual, dual_pixel_task)
        self.decoder = InstructIR_head(img_channel=inp_channels, width=out_channels, blk_nums=decoder_blk_nums, txtdim=text_dim)
    def forward(self, x, text_embed):
        x_feature = self.encoder(x)
        out = self.decoder(x_feature, text_embed)
        return out


if __name__ == '__main__':
    from torchsummary import summary
    net = DegAE_InstructIR(inp_channels=1, out_channels=64, dim=48, num_blocks=[4, 6, 6, 8], num_refinement_blocks=4, heads=[1,2,4,8], ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', dual_pixel_task=False, global_residual=False
                           ,text_dim=512, decoder_blk_nums=[2,2,2,2]).cuda()
    summary(net, input_size=[(1, 128, 128), (512,)])
