import torch
from models.unet import UNet
from models.pix2pix_networks import PixelDiscriminator
from models.liteFlownet import lite_flownet as lite_flow
from models.flownet2.models import FlowNet2SD


class convAE(torch.nn.Module):
    def __init__(self, flownet_backbone):
        super(convAE, self).__init__()
        self.generator = UNet(input_channels=12, output_channel=3)
        self.discriminator = PixelDiscriminator(input_nc=3) 
        self.flownet_backbone = flownet_backbone
        if flownet_backbone == '2sd':
            self.flow_net = FlowNet2SD()
        else:
            self.flow_net = lite_flow.Network()

    def forward(self, input_frames, target_frame, input_last):
        G_frame = self.generator(input_frames)

        if self.flownet_backbone== 'lite':
            gt_flow_input = torch.cat([input_last, target_frame], 1)
            pred_flow_input = torch.cat([input_last, G_frame], 1)
            # No need to train flow_net, use .detach() to cut off gradients.
            flow_gt = self.flow_net.batch_estimate(gt_flow_input, self.flow_net).detach()
            flow_pred = self.flow_net.batch_estimate(pred_flow_input, self.flow_net).detach()
        else:
            gt_flow_input = torch.cat([input_last.unsqueeze(2), target_frame.unsqueeze(2)], 2)
            pred_flow_input = torch.cat([input_last.unsqueeze(2), G_frame.unsqueeze(2)], 2)

            flow_gt = (self.flow_net(gt_flow_input * 255.) / 255.).detach()  # Input for flownet2sd is in (0, 255).
            flow_pred = (self.flow_net(pred_flow_input * 255.) / 255.).detach()

        return G_frame, flow_gt, flow_pred 

