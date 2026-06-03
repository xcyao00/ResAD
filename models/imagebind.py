import torch
import torch.nn as nn
import torch.nn.functional as F
from .ImageBind import *
from .ImageBind import data
import kornia as K


class ImageBindModel(nn.Module):

    def __init__(self, device='cuda:0'):
        super(ImageBindModel, self).__init__()
#DLboxで使うため
        imagebind_ckpt_path = '/home/ueno/pretrained_weights/imagebind/imagebind_huge.pth'
        
        print (f'Initializing visual encoder from {imagebind_ckpt_path} ...')
        self.visual_encoder, self.visual_hidden_size = imagebind_model.imagebind_huge({})
        imagebind_ckpt = torch.load(imagebind_ckpt_path, map_location=torch.device('cpu'))
        self.visual_encoder.load_state_dict(imagebind_ckpt, strict=True)
        # free vision encoder
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder.eval()
        print('Visual encoder initialized.')

        self.device = torch.device(device)

    def rot90_img(self, x, idx):
        degreesarr = [0., 90., 180., 270., 360]
        degrees = torch.tensor(degreesarr[idx]).half().to(self.device)
        x = K.geometry.transform.rotate(x, angle = degrees, padding_mode='reflection')
        return x
    
    def encode_image_from_image_paths(self, image_paths, device):
        inputs = {ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device)}
        # convert into visual dtype
        inputs = {key: inputs[key] for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            patch_features = embeddings['vision'][1] # bsz x h*w x 1280
            for i in range(len(patch_features)):
                patch_features[i] = patch_features[i].transpose(0, 1)[:, 1:, :]

        return patch_features
    
    def encode_image_from_tensors(self, image_tensors):
        inputs = {ModalityType.VISION: image_tensors}
        # convert into visual dtype
        inputs = {key: inputs[key] for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            patch_features = embeddings['vision'][1] # bsz x h*w x 1280
            for i in range(len(patch_features)):
                patch_features[i] = patch_features[i].transpose(0, 1)[:, 1:, :]

        return patch_features
    
    def encode_image_for_one_shot_with_aug(self, image_paths):
        image_tensors = data.load_and_transform_vision_data(image_paths, self.device).half()
        B, C, H, W = image_tensors.shape
        
        rotated_images = torch.zeros((4, B, C, H, W)).half().to(self.device)
        for j, degree in enumerate([0, 1, 2, 3]):
            rotated_img = self.rot90_img(image_tensors, degree)
            rotated_images[j] = rotated_img

        image_tensors = rotated_images.transpose(0, 1).reshape(B * 4, C, H, W)

        inputs = {ModalityType.VISION: image_tensors}
        # convert into visual dtype
        inputs = {key: inputs[key] for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            patch_features = embeddings['vision'][1] # bsz x h*w x 1280
            for i in range(len(patch_features)):
                patch_features[i] = patch_features[i].transpose(0, 1)[:, 1:, :].reshape(B, 4, 256, 1280).reshape(B, 4 * 256, 1280)

        return patch_features

    def extract_multimodal_feature(self, inputs, web_demo):
        features = []
        if inputs['image_paths']:
            if inputs['normal_image_paths']:
                # 4 layers: (1, 256, 1280)
                query_patch_embeds = self.encode_image_for_one_shot(inputs['image_paths'])
                # if 'mvtec' in inputs['normal_image_paths']:
                #     normal_patch_embeds = self.encode_image_for_one_shot_with_aug(inputs['normal_image_paths'])
                # else:
                #     # 4 layers: (1, 256, 1280)
                #     normal_patch_embeds = self.encode_image_for_one_shot(inputs['normal_image_paths'])
                normal_patch_embeds = self.encode_image_for_one_shot_with_aug(inputs['normal_image_paths'])
                
                anomaly_maps = []
                for i in range(len(query_patch_embeds)):
                    query_patch_embeds_i = query_patch_embeds[i].view(256, 1, 1280)  # (256, bs, 1280)
                    normal_patch_embeds_i = normal_patch_embeds[i].reshape(1, -1, 1280)  # (1, num_shot*256*num_aug, 1280)
                    similarity_matrix = F.cosine_similarity(query_patch_embeds_i, normal_patch_embeds_i, dim=2)
                    anomaly_map, _ = torch.max(similarity_matrix, dim=1)
                    anomaly_maps.append(anomaly_map)

                anomaly_map_mean = torch.mean(torch.stack(anomaly_maps, dim=0), dim=0).reshape(1, 1, 16, 16)
                anomaly_map_mean = F.interpolate(anomaly_map_mean, size=224, mode='bilinear', align_corners=True)
                anomaly_map_mean = 1 - anomaly_map_mean # (anomaly_map_ret + 1 - sim) / 2
           
        if inputs['audio_paths']:
            audio_embeds, _ = self.encode_audio(inputs['audio_paths'])
            features.append(audio_embeds)
        if inputs['video_paths']:
            video_embeds, _ = self.encode_video(inputs['video_paths'])
            features.append(video_embeds)
        if inputs['thermal_paths']:
            thermal_embeds, _ = self.encode_thermal(inputs['thermal_paths'])
            features.append(thermal_embeds)

        return anomaly_map_mean

    def prepare_generation_embedding(self, inputs, web_demo):
        # if len(inputs['modality_embeds']) == 1:
        #     feature_embeds = inputs['modality_embeds'][0]
        # else:
        anomaly_map = self.extract_multimodal_feature(inputs, web_demo)
    
        return anomaly_map

    def generate(self, inputs, web_demo=False):
        """inputs:
            'prompt': human input prompt,
            'image_paths': optional,
            'rimage_paths': optional,
            'audio_paths': optional
            'video_paths': optional
            'thermal_paths': optional
            'normal_image_paths': optional,
            'max_tgt_len': generation length,
            'top_p': top_p,
            'temperature': temperature
            'modality_embeds': None or torch.tensor
            'modality_cache': save the image cache
        """
        anomaly_map = self.prepare_generation_embedding(inputs, web_demo)
        
        return anomaly_map
