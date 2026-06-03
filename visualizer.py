import math
import os
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
import matplotlib.pyplot as plt
from utils import get_image_scores

def denormalization(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = np.array(mean)
    std = np.array(std)
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


class Visualizer(object):
    def __init__(self, root, prefix=''):
        self.root = root
        self.prefix = prefix
        os.makedirs(self.root, exist_ok=True)
    
    def set_prefix(self, prefix):
        self.prefix = prefix

    def plot(self, test_imgs, scores, gt_masks):
        """
        Args:
            test_imgs (ndarray): shape (N, 3, h, w)
            scores (ndarray): shape (N, h, w)
            gt_masks (ndarray): shape (N, 1, h, w)
        """
        vmax = scores.max() * 255.
        vmin = scores.min() * 255. + 80
        vmax = vmax - 20
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        img_scores = get_image_scores(scores, topk=10)#6.27
        rank = np.argsort(img_scores)
        rank = np.argsort(rank)
        for i in range(len(scores)):
            img = test_imgs[i]
            img = denormalization(img)
            gt_mask = gt_masks[i].squeeze()
            score = scores[i]
            #score = gaussian_filter(score, sigma=4)
            
            heat_map = score * 255
            fig_img, ax_img = plt.subplots(1, 3, figsize=(9, 3))

            fig_img.subplots_adjust(wspace=0.05, hspace=0)
            for ax_i in ax_img:
                ax_i.axes.xaxis.set_visible(False)
                ax_i.axes.yaxis.set_visible(False)

            ax_img[0].imshow(img)
            ax_img[0].title.set_text('Input image')
            ax_img[1].imshow(gt_mask, cmap='gray')
            ax_img[1].title.set_text('GroundTruth')
            ax_img[2].imshow(heat_map, cmap='jet', norm=norm, interpolation='none')
            ax_img[2].imshow(img, cmap='gray', alpha=0.7, interpolation='none')
            ax_img[2].title.set_text('Segmentation' + str(rank[i]) + '/' + str(len(rank)) + '/' + str(img_scores[i]))
            
            fig_img.savefig(os.path.join(self.root, str(i) + '.png'), dpi=300)
            # if img_types[i] == 'good':
            #     if img_scores[i] <= img_threshold:
            #         fig_img.savefig(os.path.join(self.root, 'normal_ok', img_types[i] + '_' + file_names[i]), dpi=300)
            #     else:
            #         fig_img.savefig(os.path.join(self.root, 'normal_nok', img_types[i] + '_' + file_names[i]), dpi=300)
            # else:
            #     if img_scores[i] > img_threshold:
            #         fig_img.savefig(os.path.join(self.root, 'anomaly_ok', img_types[i] + '_' + file_names[i]), dpi=300)
            #     else:
            #         fig_img.savefig(os.path.join(self.root, 'anomaly_nok', img_types[i] + '_' + file_names[i]), dpi=300)
              
            plt.close()
