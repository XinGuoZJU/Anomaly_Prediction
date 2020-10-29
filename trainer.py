import os
from glob import glob
import cv2
import time
import datetime
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import argparse
import random

from utils import *
from losses import *
import Dataset
from models.unet import UNet
from models.pix2pix_networks import PixelDiscriminator
from models.liteFlownet import lite_flownet as lite_flow
from config import update_config
from models.flownet2.models import FlowNet2SD
from evaluate import val



class Trainer(object):
    def __init__(self, train_dataset, model, optimizer_G, optimizer_D, data_root, train_dataloader, start_iter, flow_backbone, 
                    iters, val_interval, dataset, save_interval, train_data, batch_size, img_size, 
                    test_data, trained_model, show_curve, show_heatmap, show_flow):
        self.train_dataset = train_dataset
        self.model = model
        self.generator = model.generator
        self.discriminator = model.discriminator
        self.flow_net = model.flow_net
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.train_dataloader = train_dataloader
        self.start_iter = start_iter
        self.flow_backbone = flow_backbone
        self.show_flow = show_flow
        self.img_size = img_size

        self.iters = iters
        self.val_interval = val_interval
        self.dataset = dataset
        self.save_interval = save_interval
        self.train_data = train_data
        self.batch_size = batch_size

        self.test_data = test_data
        self.trained_model = trained_model
        self.show_curve = show_curve
        self.show_heatmap = show_heatmap

        self.data_root = data_root

        self.adversarial_loss = Adversarial_Loss().cuda()
        self.discriminate_loss = Discriminate_Loss().cuda()
        self.gradient_loss = Gradient_Loss(3).cuda()
        self.flow_loss = Flow_Loss().cuda()
        self.intensity_loss = Intensity_Loss().cuda()


    def val(self, model=None):
        if model:  # This is for testing during training.
            generator = model
            generator.eval()
        else:
            generator = UNet(input_channels=12, output_channel=3).cuda().eval()
            generator.load_state_dict(torch.load('weights/' + self.trained_model)['net_g'])
            print(f'The pre-trained generator has been loaded from \'weights/{self.trained_model}\'.\n')

        video_folders = os.listdir(self.test_data)
        video_folders.sort()
        video_folders = [os.path.join(self.test_data, aa) for aa in video_folders]

        fps = 0
        psnr_group = []

        if not model:
            if self.show_curve:
                fig = plt.figure("Image")
                manager = plt.get_current_fig_manager()
                manager.window.setGeometry(550, 200, 600, 500)
                # This works for QT backend, for other backends, check this ⬃⬃⬃.
                # https://stackoverflow.com/questions/7449585/how-do-you-set-the-absolute-position-of-figure-windows-with-matplotlib
                plt.xlabel('frames')
                plt.ylabel('psnr')
                plt.title('psnr curve')
                plt.grid(ls='--')

                cv2.namedWindow('target frames', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('target frames', 384, 384)
                cv2.moveWindow("target frames", 100, 100)

            if self.show_heatmap:
                cv2.namedWindow('difference map', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('difference map', 384, 384)
                cv2.moveWindow('difference map', 100, 550)

        with torch.no_grad():
            for i, folder in enumerate(video_folders):
                dataset = Dataset.test_dataset(self.img_size, folder)

                if not model:
                    name = folder.split('/')[-1]
                    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')

                    if self.show_curve:
                        video_writer = cv2.VideoWriter(f'results/{name}_video.avi', fourcc, 30, self.img_size)
                        curve_writer = cv2.VideoWriter(f'results/{name}_curve.avi', fourcc, 30, (600, 430))

                        js = []
                        plt.clf()
                        ax = plt.axes(xlim=(0, len(dataset)), ylim=(30, 45))
                        line, = ax.plot([], [], '-b')

                    if self.show_heatmap:
                        heatmap_writer = cv2.VideoWriter(f'results/{name}_heatmap.avi', fourcc, 30, self.img_size)

                psnrs = []
                for j, clip in enumerate(dataset):
                    input_np = clip[0:12, :, :]
                    target_np = clip[12:15, :, :]
                    input_frames = torch.from_numpy(input_np).unsqueeze(0).cuda()
                    target_frame = torch.from_numpy(target_np).unsqueeze(0).cuda()

                    G_frame = generator(input_frames)
                    test_psnr = psnr_error(G_frame, target_frame).cpu().detach().numpy()
                    psnrs.append(float(test_psnr))

                    if not model:
                        if self.show_curve:
                            cv2_frame = ((target_np + 1) * 127.5).transpose(1, 2, 0).astype('uint8')
                            js.append(j)
                            line.set_xdata(js)  # This keeps the existing figure and updates the X-axis and Y-axis data,
                            line.set_ydata(psnrs)  # which is faster, but still not perfect.
                            plt.pause(0.001)  # show curve

                            cv2.imshow('target frames', cv2_frame)
                            cv2.waitKey(1)  # show video

                            video_writer.write(cv2_frame)  # Write original video frames.

                            buffer = io.BytesIO()  # Write curve frames from buffer.
                            fig.canvas.print_png(buffer)
                            buffer.write(buffer.getvalue())
                            curve_img = np.array(Image.open(buffer))[..., (2, 1, 0)]
                            curve_writer.write(curve_img)

                        if self.show_heatmap:
                            diff_map = torch.sum(torch.abs(G_frame - target_frame).squeeze(), 0)
                            diff_map -= diff_map.min()  # Normalize to 0 ~ 255.
                            diff_map /= diff_map.max()
                            diff_map *= 255
                            diff_map = diff_map.cpu().detach().numpy().astype('uint8')
                            heat_map = cv2.applyColorMap(diff_map, cv2.COLORMAP_JET)

                            cv2.imshow('difference map', heat_map)
                            cv2.waitKey(1)

                            heatmap_writer.write(heat_map)  # Write heatmap frames.

                    torch.cuda.synchronize()
                    end = time.time()
                    if j > 1:  # Compute fps by calculating the time used in one completed iteration, this is more accurate.
                        fps = 1 / (end - temp)
                    temp = end
                    print(f'\rDetecting: [{i + 1:02d}] {j + 1}/{len(dataset)}, {fps:.2f} fps.', end='')

                psnr_group.append(np.array(psnrs))

                if not model:
                    if self.show_curve:
                        video_writer.release()
                        curve_writer.release()
                    if self.show_heatmap:
                        heatmap_writer.release()

        print('\nAll frames were detected, begin to compute AUC.')

        gt_loader = Label_loader(video_folders, self.data_root, self.dataset, self.test_data)  # Get gt labels.
        gt = gt_loader()

        assert len(psnr_group) == len(gt), f'Ground truth has {len(gt)} videos, but got {len(psnr_group)} detected videos.'

        scores = np.array([], dtype=np.float32)
        labels = np.array([], dtype=np.int8)
        for i in range(len(psnr_group)):
            distance = psnr_group[i]
            distance -= min(distance)  # distance = (distance - min) / (max - min)
            distance /= max(distance)

            scores = np.concatenate((scores, distance), axis=0)
            labels = np.concatenate((labels, gt[i][4:]), axis=0)  # Exclude the first 4 unpredictable frames in gt.

        assert scores.shape == labels.shape, \
            f'Ground truth has {labels.shape[0]} frames, but got {scores.shape[0]} detected frames.'

        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
        auc = metrics.auc(fpr, tpr)
        print(f'AUC: {auc}\n')
        return auc


    def train(self):
        step = self.start_iter
        writer = SummaryWriter(f'tensorboard_log/{self.dataset}_bs{self.batch_size}')
        training = True
        self.generator.train()
        self.discriminator.train()
        
        for indice, clips, flow_strs in self.train_dataloader:
            print()
            print(step)
            input_frames = clips[:, 0:12, :, :].cuda()  # (n, 12, 256, 256)
            target_frame = clips[:, 12:15, :, :].cuda()  # (n, 3, 256, 256)
            input_last = input_frames[:, 9:12, :, :].cuda()  # use for flow_loss

            ## pop() the used frame index, this can't work in train_dataset.__getitem__ because of multiprocessing.
            #for index in indice:
            #    self.train_dataset.all_seqs[index].pop()
            #    if len(self.train_dataset.all_seqs[index]) == 0:
            #        self.train_dataset.all_seqs[index] = list(range(len(train_dataset.videos[index]) - 4))
            #        random.shuffle(self.train_dataset.all_seqs[index])
            print(input_frames)
            G_frame, flow_gt, flow_pred = self.model.forward(input_frames, target_frame, input_last)

            if self.show_flow:
                flow = np.array(flow_gt.cpu().detach().numpy().transpose(0, 2, 3, 1), np.float32)  # to (n, w, h, 2)
                for i in range(flow.shape[0]):
                    aa = flow_to_color(flow[i], convert_to_bgr=False)
                    path = self.train_data.split('/')[-3] + '_' + flow_strs[i]
                    cv2.imwrite(f'images/{path}.jpg', aa)  # e.g. images/avenue_4_574-575.jpg
                    print(f'Saved a sample optic flow image from gt frames: \'images/{path}.jpg\'.')

            inte_l = self.intensity_loss(G_frame, target_frame)
            grad_l = self.gradient_loss(G_frame, target_frame)
            fl_l = self.flow_loss(flow_pred, flow_gt)
            g_l = self.adversarial_loss(self.discriminator(G_frame))
            G_l_t = 1. * inte_l + 1. * grad_l + 2. * fl_l + 0.05 * g_l

            # When training discriminator, don't train generator, so use .detach() to cut off gradients.
            D_l = self.discriminate_loss(self.discriminator(target_frame), self.discriminator(G_frame.detach()))

            # https://github.com/pytorch/pytorch/issues/39141
            # torch.optim optimizer now do inplace detection for module parameters since PyTorch 1.5
            # If I do this way:
            # ----------------------------------------
            # optimizer_D.zero_grad()
            # D_l.backward()
            # optimizer_D.step()
            # optimizer_G.zero_grad()
            # G_l_t.backward()
            # optimizer_G.step()
            # ----------------------------------------
            # The optimizer_D.step() modifies the discriminator parameters inplace.
            # But these parameters are required to compute the generator gradient for the generator.

            # Thus I should make sure no parameters are modified before calling .step(), like this:
            # ----------------------------------------
            # optimizer_G.zero_grad()
            # G_l_t.backward()
            # optimizer_G.step()
            # optimizer_D.zero_grad()
            # D_l.backward()
            # optimizer_D.step()
            # ----------------------------------------

            # Or just do .step() after all the gradients have been computed, like the following way:
            self.optimizer_D.zero_grad()
            D_l.backward()
            self.optimizer_G.zero_grad()
            G_l_t.backward()
            self.optimizer_D.step()
            self.optimizer_G.step()

            torch.cuda.synchronize()
            time_end = time.time()
            if step > self.start_iter:  # This doesn't include the testing time during training.
                iter_t = time_end - temp
            temp = time_end

            if step != self.start_iter:
                if step % 20 == 0:
                    time_remain = (self.iters - step) * iter_t
                    eta = str(datetime.timedelta(seconds=time_remain)).split('.')[0]
                    psnr = psnr_error(G_frame, target_frame)
                    lr_g = self.optimizer_G.param_groups[0]['lr']
                    lr_d = self.optimizer_D.param_groups[0]['lr']

                    print(f"[{step}]  inte_l: {inte_l:.3f} | grad_l: {grad_l:.3f} | fl_l: {fl_l:.3f} | "
                          f"g_l: {g_l:.3f} | G_l_total: {G_l_t:.3f} | D_l: {D_l:.3f} | psnr: {psnr:.3f} | "
                          f"iter: {iter_t:.3f}s | ETA: {eta} | lr: {lr_g} {lr_d}")

                    save_G_frame = ((G_frame[0] + 1) / 2)
                    save_G_frame = save_G_frame.cpu().detach()[(2, 1, 0), ...]
                    save_target = ((target_frame[0] + 1) / 2)
                    save_target = save_target.cpu().detach()[(2, 1, 0), ...]

                    writer.add_scalar('psnr/train_psnr', psnr, global_step=step)
                    writer.add_scalar('total_loss/g_loss_total', G_l_t, global_step=step)
                    writer.add_scalar('total_loss/d_loss', D_l, global_step=step)
                    writer.add_scalar('G_loss_total/g_loss', g_l, global_step=step)
                    writer.add_scalar('G_loss_total/fl_loss', fl_l, global_step=step)
                    writer.add_scalar('G_loss_total/inte_loss', inte_l, global_step=step)
                    writer.add_scalar('G_loss_total/grad_loss', grad_l, global_step=step)
                    writer.add_scalar('psnr/train_psnr', psnr, global_step=step)

                if step % int(self.iters / 100) == 0:
                    writer.add_image('image/G_frame', save_G_frame, global_step=step)
                    writer.add_image('image/target', save_target, global_step=step)

                if step % self.save_interval == 0:
                    model_dict = {'net_g': self.generator.state_dict(), 'optimizer_g': self.optimizer_G.state_dict(),
                                  'net_d': self.discriminator.state_dict(), 'optimizer_d': self.optimizer_D.state_dict()}
                    torch.save(model_dict, f'weights/{self.dataset}_{step}.pth')
                    print(f'\nAlready saved: \'{self.dataset}_{step}.pth\'.')

                if step % self.val_interval == 0:
                    auc = val(model=self.generator)
                    writer.add_scalar('results/auc', auc, global_step=step)
                    self.generator.train()

            step += 1
            if step > self.iters:
                training = False
                model_dict = {'net_g': self.generator.state_dict(), 'optimizer_g': self.optimizer_G.state_dict(),
                              'net_d': self.discriminator.state_dict(), 'optimizer_d': self.optimizer_D.state_dict()}
                torch.save(model_dict, f'weights/latest_{self.dataset}_{step}.pth')
                print('gx12313213')
                break


