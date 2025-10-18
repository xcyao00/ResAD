import os
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
import timm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.mvtec import MVTEC
from datasets.visa import VISA
from datasets.btad import BTAD
from datasets.mvtec_3d import MVTEC3D
from datasets.mvtec_loco import MVTECLOCO
from datasets.mpdd import MPDD
from scipy.stats import entropy


device = 'cuda:0'
S = 5  # grid size
NC = 12  # Number of clustering centers
K = 10  # number of KNN samples
tau = 5  # tau in E.q(5)
img_size = 320

root_dir = '/path/to/your/dataset'
DATASET_CLASS = MVTEC
DATASET_NAME = 'mvtec'

def get_reference_features(encoder, class_name):
    train_dataset = DATASET_CLASS(root_dir, class_name=class_name, train=True, img_size=img_size, crp_size=img_size,
                          msk_size=img_size, msk_crp_size=img_size)
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=8, drop_last=False
    )
    
    progress_bar = tqdm(total=len(train_loader))
    progress_bar.set_description(f"Extract Features {class_name}")
    features = []
    for step, batch in enumerate(train_loader):
        progress_bar.update(1)
        images, _, _, _, _ = batch
        
        images = images.to(device)
        
        with torch.no_grad():
            feature = encoder(images)
        features.append(feature[0])
    progress_bar.close()
    
    features = torch.cat(features, dim=0).permute(0, 2, 3, 1)
    NR, HP, WP, CP = features.shape
    features = features.reshape(NR * HP * WP, CP)
    
    # For avoid OpenBLAS warning: precompiled NUM_THREADS exceeded
    # vim ~/.bashrc
    # export OMP_NUM_THREADS=1
    # export OPENBLAS_NUM_THREADS=1
    # source ~/.bashrc
    
    features = features.cpu().numpy()
    print('feature size : ', features.shape)
    print("fitting reference features...")
    kmeans = KMeans(n_clusters=NC, n_init='auto', random_state=0)
    kmeans.fit(features)
    ref_features = kmeans.cluster_centers_
    print('feature size :', ref_features.shape)
    
    os.makedirs(os.path.join("references", "centers", DATASET_NAME), exist_ok=True)
    np.save(os.path.join(f"references/centers/{DATASET_NAME}", f"{class_name}_centers.npy"), ref_features)
    
    return ref_features


def load_reference_features(class_name):
    path = os.path.join(f"references/centers/{DATASET_NAME}", f"{class_name}_centers.npy")
    ref_features = np.load(path)
    ref_features = torch.from_numpy(ref_features).to(device)
    
    return ref_features


def get_reference_bow_histograms(encoder, class_name, ref_features):
    """
    Get reference bow histograms for all reference samples.
    """
    train_dataset = DATASET_CLASS(root_dir, class_name=class_name, train=True, img_size=img_size, crp_size=img_size,
                          msk_size=img_size, msk_crp_size=img_size)
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False
    )
    
    all_histograms = []
    progress_bar = tqdm(total=len(train_loader))
    progress_bar.set_description(f"Get Reference Histograms")
    for _, batch in enumerate(train_loader):
        progress_bar.update(1)
        images, _, _, _, _ = batch
        
        images = images.to(device)
        
        with torch.no_grad():
            feature = encoder(images)
            feature = feature[0]
            bow_map = get_bow_map(feature, ref_features)
            bow_map = bow_map.cpu().numpy()
            histograms = get_bow_histograms(bow_map)
            all_histograms.append(histograms)
    progress_bar.close()
    
    return all_histograms


def get_bow_map(features, ref_features):
    """
    Args:
        features (Tensor): (1, C, H, W).
        ref_features (Tensor): (Nc, C).
    """
    _, CP, HP, WP = features.shape
    features = features.permute(0, 2, 3, 1).reshape(HP * WP, CP)
    features = features.unsqueeze(1)
    ref_features = ref_features.unsqueeze(0)
    cosine_matrix = F.cosine_similarity(features, ref_features, dim=-1)  # (Hp * Wp, Nc)
    indices = torch.argmax(cosine_matrix, dim=1)
    bow_map = indices.reshape(HP, WP)
    
    return bow_map


def get_bow_histogram(bows):
    histogram = [0.0 for _ in range(NC)]
    N = bows.shape[0]
    for i in range(NC):
        histogram[i] = np.sum(bows == i) / N
    return np.array(histogram)


def get_bow_histograms(bow_map):
    HP, WP = bow_map.shape
    # number of sub grids in height and width 
    H_sub, W_sub = HP // S, WP // S  
    
    histograms = []
    # for each sub grid
    for i in range(S):
        for j in range(S):
            bows = bow_map[i*H_sub:(i+1)*H_sub, j*W_sub:(j+1)*W_sub]  # bag of words in this grid
            bows = bows.reshape(-1)
            histogram = get_bow_histogram(bows)
            histograms.append(histogram)
    return histograms


def calculate_global_distance(histograms, ref_histograms, tau=5):
    S_2 = len(histograms)
    
    def KL(ref_histogram, histogram):
        return entropy(ref_histogram, histogram)
        
    # calculating all KL divergences
    kl_divergences = []
    for i in range(S_2):
        # the histogram in th (u, v) position
        histogram_uv, ref_histogram_uv = histograms[i], ref_histograms[i]
        kl_div = KL(ref_histogram_uv, histogram_uv)
        kl_divergences.append(kl_div)
    
    sorted_kl_divergences = np.sort(kl_divergences)
    # calculating the global distance
    D_global = 0
    for j in range(S_2 - tau): 
        D_global += sorted_kl_divergences[j]
    D_global /= (S_2 - tau)
    
    return D_global
    
    
def get_aligned_reference_samples(encoder, class_name):
    test_dataset = DATASET_CLASS(root_dir, class_name=class_name, train=False, img_size=img_size, crp_size=img_size,
                          msk_size=img_size, msk_crp_size=img_size)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False
    )
    
    # clustering centers
    ref_features = load_reference_features(class_name)
    # histograms for all reference samples
    all_ref_histograms = get_reference_bow_histograms(encoder, class_name, ref_features)
    
    progress_bar = tqdm(total=len(test_loader))
    progress_bar.set_description(f"Extract Features")
    all_knn_indices = []
    for _, batch in enumerate(test_loader):
        progress_bar.update(1)
        images, _, _, _, _ = batch
        
        images = images.to(device)
        
        with torch.no_grad():
            feature = encoder(images)
            feature = feature[0]
            
        bow_map = get_bow_map(feature, ref_features)
        bow_map = bow_map.cpu().numpy()
        histograms = get_bow_histograms(bow_map)
        # calculating global distances with reference samples
        global_distances = []
        for i in range(len(all_ref_histograms)):
            ref_histograms = all_ref_histograms[i]
            global_distance = calculate_global_distance(histograms, ref_histograms)
            global_distances.append(global_distance)
        # argpartition: the first Kth are the K minimum values
        #knn_indices = np.argpartition(global_distances, K)[:K]
        global_distances = np.array(global_distances)
        knn_indices = np.argsort(global_distances)[:K]
        all_knn_indices.append(knn_indices)
    progress_bar.close()
    all_knn_indices = np.stack(all_knn_indices, axis=0) # (N_test, K)
    
    os.makedirs(os.path.join("references", "indices10", DATASET_NAME), exist_ok=True)
    np.save(os.path.join(f"references/indices10/{DATASET_NAME}", f"{class_name}_knn_indices.npy"), all_knn_indices)

      
if __name__ == '__main__':
    os.environ['OPENBLAS_NUM_THREADS'] = '1'  # For avoid OpenBLAS warning: precompiled NUM_THREADS exceeded
    encoder = timm.create_model("densenet201", features_only=True, 
            out_indices=(1, ), pretrained=True).eval()
    encoder = encoder.to(device)
    for class_name in DATASET_CLASS.CLASS_NAMES:
        # first step
        get_reference_features(encoder, class_name)
        # second step
        # get_aligned_reference_samples(encoder, class_name)
    

    
    
            
