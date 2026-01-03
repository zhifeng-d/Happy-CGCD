import argparse
import os
import math
import time
from tqdm import tqdm
from copy import deepcopy

from sklearn.cluster import KMeans
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from project_utils.general_utils import set_seed
from project_utils.general_utils import init_experiment, AverageMeter
from project_utils.cluster_and_log_utils import log_accs_from_preds

from data.augmentations import get_transform
from data.get_datasets import get_class_splits, ContrastiveLearningViewGenerator, get_datasets

from models import vision_transformer as vits

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import torch.nn.functional as F
# -----------------
# OTHER PATHS
# -----------------
dino_pretrain_path = '/home/asus/GPC-master/checkpoints/dino_vitbase16_pretrain.pth'
feature_extract_dir = '/lustre/home/sjma/Continual-GCD-project/GCD-baseline/extracted_features_public_impl'     # Extract features to this directory
exp_root_happy = '/home/asus/Happy-CGCD3/dev_outputs_Happy'  


'''
2024-05-03
prototype augmentation utils, including:
* prototype augmentation loss functions
* save prototypes for the offline session (according to ground-truth labels)
* save prototypes for each online continual session (according to pseudo-labels)

update 1: [2024-05-04] hardness-aware prototype sampling
'''


'''====================================================================================================================='''
'''SOP: Selective Old-Class Protection Module'''
'''====================================================================================================================='''
class SelectiveOldClassProtection:
    """
    基于簇置信度的选择性旧类保护模块
    
    核心思想:
    - 不是"是否保护旧类",而是"保护哪些旧类、保护到什么程度"
    - 高稳定旧类 → 强保护
    - 低稳定旧类 → 弱保护(允许适配)
    """
    
    def __init__(self, num_old_classes, lambda_max=1.0, lambda_min=0.1, 
                 stability_momentum=0.9, use_entropy=True):
        """
        Args:
            num_old_classes: 旧类的数量
            lambda_max: 最大正则化强度 (高稳定类)
            lambda_min: 最小正则化强度 (低稳定类)
            stability_momentum: 稳定性得分的动量更新系数
            use_entropy: 是否使用熵作为稳定性度量(False则使用最大置信度)
        """
        self.num_old_classes = num_old_classes
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min
        self.stability_momentum = stability_momentum
        self.use_entropy = use_entropy
        
        # 存储每个旧类的稳定性得分
        self.stability_scores = None
        # 存储每个旧类的正则化权重
        self.lambda_weights = None
        
    def compute_stability_scores(self, model, data_loader, device):
        """
        计算每个旧类的稳定性得分
        
        Args:
            model: 当前模型
            data_loader: 数据加载器
            device: 设备
            
        Returns:
            stability_scores: shape (num_old_classes,) 每个旧类的稳定性得分
        """
        model.eval()
        
        # 累积每个类别的置信度/熵
        class_confidence_sum = torch.zeros(self.num_old_classes).to(device)
        class_entropy_sum = torch.zeros(self.num_old_classes).to(device)
        class_count = torch.zeros(self.num_old_classes).to(device)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if len(batch) == 4:  # online training format
                    images, class_labels, uq_idxs, _ = batch
                else:  # offline training format
                    images, class_labels, uq_idxs = batch
                    
                images = torch.cat(images, dim=0).cuda(non_blocking=True)
                
                _, logits = model(images)
                probs = torch.softmax(logits / 0.1, dim=-1)
                
                # 只考虑旧类的输出
                old_probs = probs[:, :self.num_old_classes]
                
                # 计算每个样本对每个旧类的置信度
                max_probs, pred_classes = old_probs.max(dim=1)
                
                # 计算熵
                entropy = -(old_probs * (old_probs + 1e-10).log()).sum(dim=1)
                
                # 累积统计
                for k in range(self.num_old_classes):
                    mask = (pred_classes == k)
                    if mask.sum() > 0:
                        class_confidence_sum[k] += max_probs[mask].sum()
                        class_entropy_sum[k] += entropy[mask].sum()
                        class_count[k] += mask.sum()
        
        # 计算平均值
        class_count = torch.clamp(class_count, min=1.0)  # 避免除零
        avg_confidence = class_confidence_sum / class_count
        avg_entropy = class_entropy_sum / class_count
        
        # 根据选择使用置信度或熵作为稳定性度量
        if self.use_entropy:
            # 熵越低 → 稳定性越高
            # 归一化到 [0, 1], 其中1表示最稳定
            max_entropy = avg_entropy.max()
            min_entropy = avg_entropy.min()
            if max_entropy > min_entropy:
                stability = 1.0 - (avg_entropy - min_entropy) / (max_entropy - min_entropy + 1e-10)
            else:
                stability = torch.ones_like(avg_entropy)
        else:
            # 置信度越高 → 稳定性越高
            # 直接使用平均置信度作为稳定性
            stability = avg_confidence
        
        # 动量更新稳定性得分
        if self.stability_scores is None:
            self.stability_scores = stability
        else:
            self.stability_scores = (self.stability_momentum * self.stability_scores + 
                                   (1 - self.stability_momentum) * stability)
        
        return self.stability_scores
    
    def compute_lambda_weights(self):
        """
        根据稳定性得分计算每个旧类的正则化权重
        
        Returns:
            lambda_weights: shape (num_old_classes,) 每个旧类的正则化权重
        """
        if self.stability_scores is None:
            # 如果还没有计算稳定性得分,使用均匀权重
            self.lambda_weights = torch.ones(self.num_old_classes) * \
                                 (self.lambda_max + self.lambda_min) / 2
        else:
            # lambda(s_k) = lambda_min + (lambda_max - lambda_min) * s_k
            self.lambda_weights = (self.lambda_min + 
                                 (self.lambda_max - self.lambda_min) * self.stability_scores)
        
        return self.lambda_weights
    
    def compute_sop_loss(self, model_cur, model_prev):
        """
        计算选择性旧类保护损失
        
        L_sop = sum_{k in C_old} lambda(s_k) * ||w_k - w_k^prev||^2
        
        Args:
            model_cur: 当前模型
            model_prev: 上一阶段的模型
            
        Returns:
            sop_loss: 选择性旧类保护损失
        """
        if self.lambda_weights is None:
            self.compute_lambda_weights()
        
        # 获取当前和之前的分类器权重
        w_cur = model_cur[1].last_layer.weight[:self.num_old_classes]  # (num_old_classes, feat_dim)
        w_prev = model_prev[1].last_layer.weight[:self.num_old_classes]  # (num_old_classes, feat_dim)
        
        # 计算每个类的权重偏移
        weight_diff = (w_cur - w_prev).pow(2).sum(dim=1)  # (num_old_classes,)
        
        # 应用类别特定的正则化权重
        lambda_weights = self.lambda_weights.to(weight_diff.device)
        sop_loss = (lambda_weights * weight_diff).mean()
        
        return sop_loss


class ProtoAugManager:
    def __init__(self, feature_dim, batch_size, hardness_temp, radius_scale, device, logger):
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.device = device
        self.prototypes = None
        self.mean_similarity = None   # NOTE!!! mean similarity of each prototype, for hardness-aware sampling
        self.hardness_temp = hardness_temp   # NOTE!!! temperature to compute mean similarity to softmax prob for hardness-aware sampling
        self.radius = 0
        self.radius_scale = radius_scale
        self.logger = logger


    def save_proto_aug_dict(self, save_path):
        proto_aug_dict = {
            'prototypes': self.prototypes,
            'radius': self.radius,
            'mean_similarity': self.mean_similarity,
        }

        torch.save(proto_aug_dict, save_path)

    # load continual   # NOTE!!!
    def load_proto_aug_dict(self, load_path):
        proto_aug_dict = torch.load(load_path)

        self.prototypes = proto_aug_dict['prototypes']
        self.radius = proto_aug_dict['radius']
        self.mean_similarity = proto_aug_dict['mean_similarity']


    def compute_proto_aug_loss(self, model):
        prototypes = F.normalize(self.prototypes, dim=-1, p=2).to(self.device)
        prototypes_labels = torch.randint(0, len(prototypes), (self.batch_size,)).to(self.device)   # dtype=torch.long
        prototypes_sampled = prototypes[prototypes_labels]
        prototypes_augmented = prototypes_sampled + torch.randn((self.batch_size, self.feature_dim), device=self.device) * self.radius * self.radius_scale
        #prototypes_augmented = F.normalize(prototypes_augmented, dim=-1, p=2) # NOTE!!! DO NOT normalize
        # forward prototypes and get logits
        _, prototypes_output = model[1](prototypes_augmented)
        proto_aug_loss = nn.CrossEntropyLoss()(prototypes_output / 0.1, prototypes_labels)

        return proto_aug_loss


    def compute_proto_aug_hardness_aware_loss(self, model):
        prototypes = F.normalize(self.prototypes, dim=-1, p=2).to(self.device)

        # hardness-aware sampling
        sampling_prob = F.softmax(self.mean_similarity / self.hardness_temp, dim=-1)
        sampling_prob = sampling_prob.cpu().numpy()
        prototypes_labels = np.random.choice(len(prototypes), size=(self.batch_size,), replace=True, p=sampling_prob)
        prototypes_labels = torch.from_numpy(prototypes_labels).long().to(self.device)

        prototypes_sampled = prototypes[prototypes_labels]
        prototypes_augmented = prototypes_sampled + torch.randn((self.batch_size, self.feature_dim), device=self.device) * self.radius * self.radius_scale
        #prototypes_augmented = F.normalize(prototypes_augmented, dim=-1, p=2) # NOTE!!! DO NOT normalize
        # forward prototypes and get logits
        _, prototypes_output = model[1](prototypes_augmented)
        proto_aug_loss = nn.CrossEntropyLoss()(prototypes_output / 0.1, prototypes_labels)

        return proto_aug_loss


    def update_prototypes_offline(self, model, train_loader, num_labeled_classes):
        model.eval()

        all_feats_list = []
        all_labels_list = []
        # forward data
        for batch_idx, (images, label, _) in enumerate(tqdm(train_loader)):   # NOTE!!!
            images = images.cuda(non_blocking=True)
            with torch.no_grad():
                feats = model[0](images)   # backbone
                feats = torch.nn.functional.normalize(feats, dim=-1)
                all_feats_list.append(feats)
                all_labels_list.append(label)
        all_feats = torch.cat(all_feats_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0)

        # compute prototypes and radius
        prototypes_list = []
        radius_list = []
        for c in range(num_labeled_classes):
            feats_c = all_feats[all_labels==c]
            feats_c_mean = torch.mean(feats_c, dim=0)
            prototypes_list.append(feats_c_mean)
            feats_c_center = feats_c - feats_c_mean
            cov = torch.matmul(feats_c_center.t(), feats_c_center) / len(feats_c_center)
            radius = torch.trace(cov) / self.feature_dim   # or feats_c_center.shape[1]
            radius_list.append(radius)
        avg_radius = torch.sqrt(torch.mean(torch.stack(radius_list)))
        prototypes_all = torch.stack(prototypes_list, dim=0)
        prototypes_all = F.normalize(prototypes_all, dim=-1, p=2)

        # update
        self.radius = avg_radius
        self.prototypes = prototypes_all

        # update mean similarity for each prototype
        similarity = prototypes_all @ prototypes_all.T
        for i in range(len(similarity)):
            similarity[i,i] -= similarity[i,i]
        mean_similarity = torch.sum(similarity, dim=-1) / (len(similarity)-1)

        self.mean_similarity = mean_similarity


    def update_prototypes_online(self, model, train_loader, num_seen_classes, num_all_classes):
        model.eval()

        all_preds_list = []
        all_feats_list = []
        # forward data
        for batch_idx, (images, label, _, _) in enumerate(tqdm(train_loader)):   # NOTE!!!
            images = images.cuda(non_blocking=True)
            with torch.no_grad():
                _, logits = model(images)
                feats = model[0](images)   # backbone
                feats = torch.nn.functional.normalize(feats, dim=-1)
                all_feats_list.append(feats)
                all_preds_list.append(logits.argmax(1))
        all_feats = torch.cat(all_feats_list, dim=0)
        all_preds = torch.cat(all_preds_list, dim=0)

        # compute prototypes
        prototypes_list = []
        for c in range(num_seen_classes, num_all_classes):
            feats_c = all_feats[all_preds==c]
            if len(feats_c) == 0:
                self.logger.info('No pred of this class, using fc (last_layer) parameters...')
                feats_c_mean = model[1].last_layer.weight_v.data[c]
            else:
                self.logger.info('computing (predicted) class-wise mean...')
                feats_c_mean = torch.mean(feats_c, dim=0)
            prototypes_list.append(feats_c_mean)
        prototypes_cur = torch.stack(prototypes_list, dim=0)   # NOTE!!!
        prototypes_all = torch.cat([self.prototypes, prototypes_cur], dim=0)
        prototypes_all = F.normalize(prototypes_all, dim=-1, p=2)

        # update
        self.prototypes = prototypes_all

        # update mean similarity for each prototype
        similarity = prototypes_all @ prototypes_all.T
        for i in range(len(similarity)):
            similarity[i,i] -= similarity[i,i]
        mean_similarity = torch.sum(similarity, dim=-1) / (len(similarity)-1)

        self.mean_similarity = mean_similarity

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_proj = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        # x = x.detach()
        logits = self.last_layer(x)
        return x_proj, logits
def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]
class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
def info_nce_logits(features, n_views=2, temperature=1.0, device='cuda'):

    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels
class DistillLoss(nn.Module):
    def __init__(self, warmup_teacher_temp_epochs, nepochs, 
                 ncrops=2, warmup_teacher_temp=0.07, teacher_temp=0.04,
                 student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax(teacher_output / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss
def get_kmeans_centroid_for_new_head(model, online_session_train_loader, args, device):

    model.to(device)
    model.eval()

    all_feats = []

    args.logger.info('Perform KMeans for new classification head initialization!')
    args.logger.info('Collating features...')
    # First extract all features
    with torch.no_grad():
        for batch_idx, (images, label, _, _) in enumerate(tqdm(online_session_train_loader)):
            images = images.cuda(non_blocking=True)
            # Pass features through base model and then additional learnable transform (linear layer)
            feats = model[0](images)   # backbone
            feats = torch.nn.functional.normalize(feats, dim=-1)
            all_feats.append(feats.cpu().numpy())

    # -----------------------
    # K-MEANS
    # -----------------------
    print('Fitting K-Means...')
    all_feats = np.concatenate(all_feats)
    kmeans = KMeans(n_clusters=args.num_labeled_classes+args.num_cur_novel_classes, random_state=0).fit(all_feats)
    #preds = kmeans.labels_
    centroids_np = kmeans.cluster_centers_   # (60, 768)
    print('Done!')

    centroids = torch.from_numpy(centroids_np).to(device)
    centroids = torch.nn.functional.normalize(centroids, dim=-1)   # torch.Size([60, 768])
    #centroids = centroids.float()
    with torch.no_grad():
        _, logits = model[1](centroids)   # torch.Size([60, 50])
        max_logits, _ = torch.max(logits, dim=-1)   # torch.Size([60])
        _, proto_idx = torch.topk(max_logits, k=args.num_novel_class_per_session, largest=False)   # torch.Size([10])
        new_head = centroids[proto_idx]   # torch.Size([10, 768])

    return new_head

'''offline train and test'''
'''====================================================================================================================='''
def train_offline(student, train_loader, test_loader, args):

    params_groups = get_params_groups(student)
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs_offline,
            eta_min=args.lr * 1e-3,
        )

    cluster_criterion = DistillLoss(
                        args.warmup_teacher_temp_epochs,
                        args.epochs_offline,
                        args.n_views,
                        args.warmup_teacher_temp,
                        args.teacher_temp,
                    )
    # best acc log
    best_test_acc_old = 0

    for epoch in range(args.epochs_offline):
        loss_record = AverageMeter()

        student.train()
        for batch_idx, batch in enumerate(train_loader):

            images, class_labels, uq_idxs = batch   # NOTE!!! no mask lab in this setting
            mask_lab = torch.ones_like(class_labels)   # NOTE!!! all samples are labeled

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            student_proj, student_out = student(images)
            teacher_out = student_out.detach()

            # clustering, sup
            sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
            sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
            cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

            # clustering, unsup
            cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
            avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
            me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
            cluster_loss += args.memax_weight * me_max_loss

            # represent learning, unsup
            contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
            contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

            # representation learning, sup
            student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
            student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
            sup_con_labels = class_labels[mask_lab]
            sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)

            # Total loss
            loss = 0
            loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
            loss += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss

            # logs
            pstr = ''
            pstr += f'cls_loss: {cls_loss.item():.4f} '
            pstr += f'cluster_loss: {cluster_loss.item():.4f} '
            pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
            pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '

            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                            .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        args.logger.info('Testing on disjoint test set...')
        all_acc_test, old_acc_test, _ = test_offline(student, test_loader, epoch=epoch, save_name='Test ACC', args=args)
        args.logger.info('Test Accuracies: All {:.4f} | Old {:.4f}'.format(all_acc_test, old_acc_test))

        # Step schedule
        exp_lr_scheduler.step()

        save_dict = {
            'model': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }

        torch.save(save_dict, args.model_path)
        args.logger.info("model saved to {}.".format(args.model_path))

        if old_acc_test > best_test_acc_old:

            args.logger.info(f'Best ACC on Old Classes on test set: {old_acc_test:.4f}...')

            torch.save(save_dict, args.model_path[:-3] + f'_best.pt')
            args.logger.info("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))

            best_test_acc_old = old_acc_test

        args.logger.info(f'Exp Name: {args.exp_name}')
        args.logger.info(f'Metrics with best model on test set: Old: {best_test_acc_old:.4f}')
        args.logger.info('\n')


def test_offline(model, test_loader, epoch, save_name, args):

    model.eval()

    preds, targets = [], []
    mask = np.array([])
    # First extract all features
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, logits = model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    return all_acc, old_acc, new_acc
'''====================================================================================================================='''




'''online train and test'''
'''====================================================================================================================='''
def train_online(student, student_pre, proto_aug_manager, sop_module, train_loader, test_loader, current_session, args):

    params_groups = get_params_groups(student)
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs_online_per_session,
            eta_min=args.lr * 1e-3,
        )

    # 确保 warmup epochs 不超过总 epochs
    warmup_epochs = min(args.warmup_teacher_temp_epochs, args.epochs_online_per_session - 1)
    if warmup_epochs < args.warmup_teacher_temp_epochs:
        args.logger.info(f'Warning: warmup_teacher_temp_epochs ({args.warmup_teacher_temp_epochs}) >= epochs_online_per_session ({args.epochs_online_per_session})')
        args.logger.info(f'Adjusted warmup_teacher_temp_epochs to {warmup_epochs}')

    cluster_criterion = DistillLoss(
                        warmup_epochs,
                        args.epochs_online_per_session,
                        args.n_views,
                        args.warmup_teacher_temp,
                        args.teacher_temp,
                    )

    # best acc log
    best_test_acc_all = 0
    best_test_acc_old = 0
    best_test_acc_new = 0

    best_test_acc_soft_all = 0
    best_test_acc_seen = 0
    best_test_acc_unseen = 0

    device = torch.device('cuda:0')

    for epoch in range(args.epochs_online_per_session):
        loss_record = AverageMeter()

        # 在每个epoch开始时更新稳定性得分
        if epoch % args.sop_update_freq == 0 and current_session >= 1 and args.sop_weight > 0:
            args.logger.info('Computing stability scores for old classes...')
            sop_module.compute_stability_scores(student, train_loader, device)
            sop_module.compute_lambda_weights()
            
            # 记录稳定性得分和权重
            if args.sop_log_stability:
                stability_str = 'Stability scores: ' + ' '.join([f'{s:.3f}' for s in sop_module.stability_scores.cpu().numpy()])
                lambda_str = 'Lambda weights: ' + ' '.join([f'{l:.3f}' for l in sop_module.lambda_weights.cpu().numpy()])
                args.logger.info(stability_str)
                args.logger.info(lambda_str)

        student.train()
        student_pre.eval()
        for batch_idx, batch in enumerate(train_loader):

            images, class_labels, uq_idxs, _ = batch   # NOTE!!!   mask lab in this setting
            mask_lab = torch.zeros_like(class_labels)   # NOTE!!! all samples are unlabeled

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            student_proj, student_out = student(images)
            teacher_out = student_out.detach()

            # clustering, unsup
            cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
            avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
            #me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
            #cluster_loss += args.memax_weight * me_max_loss

            # 1. inter old and new
            avg_probs_old_in = avg_probs[:args.num_seen_classes]
            avg_probs_new_in = avg_probs[args.num_seen_classes:]

            #avg_probs_old_new = torch.tensor([torch.sum(avg_probs_old_in), torch.sum(avg_probs_new_in)], requires_grad=True, device=device)
            #me_max_loss_old_new = - torch.sum(torch.log(avg_probs_old_new**(-avg_probs_old_new))) + math.log(float(len(avg_probs_old_new)))
            avg_probs_old_marginal, avg_probs_new_marginal = torch.sum(avg_probs_old_in), torch.sum(avg_probs_new_in)
            me_max_loss_old_new =  avg_probs_old_marginal * torch.log(avg_probs_old_marginal) + avg_probs_new_marginal * torch.log(avg_probs_new_marginal) + math.log(2)

            # 2. old (intra) & new (intra)
            avg_probs_old_in_norm = avg_probs_old_in / torch.sum(avg_probs_old_in)   # norm
            avg_probs_new_in_norm = avg_probs_new_in / torch.sum(avg_probs_new_in)   # norm
            me_max_loss_old_in = - torch.sum(torch.log(avg_probs_old_in_norm**(-avg_probs_old_in_norm))) + math.log(float(len(avg_probs_old_in_norm)))
            if args.num_novel_class_per_session > 1:
                me_max_loss_new_in = - torch.sum(torch.log(avg_probs_new_in_norm**(-avg_probs_new_in_norm))) + math.log(float(len(avg_probs_new_in_norm)))
            else:
                me_max_loss_new_in = torch.tensor(0.0, device=device)
            # overall me-max loss
            cluster_loss += args.memax_old_new_weight * me_max_loss_old_new + \
                args.memax_old_in_weight * me_max_loss_old_in + args.memax_new_in_weight * me_max_loss_new_in


            # represent learning, unsup
            contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
            contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

            proto_aug_loss = proto_aug_manager.compute_proto_aug_hardness_aware_loss(student)
            feats = student[0](images)
            feats = torch.nn.functional.normalize(feats, dim=-1)
            with torch.no_grad():
                feats_pre = student_pre[0](images)
                feats_pre = torch.nn.functional.normalize(feats_pre, dim=-1)
            feat_distill_loss = (feats-feats_pre).pow(2).sum() / len(feats)

            # ========================================
            # SOP Loss: Selective Old-Class Protection
            # ========================================
            if current_session >= 1 and args.sop_weight > 0:
                sop_loss = sop_module.compute_sop_loss(student, student_pre)
            else:
                sop_loss = torch.tensor(0.0, device=device)

            # Total loss
            loss = 0
            loss += 1 * cluster_loss
            loss += 1 * contrastive_loss
            loss += args.proto_aug_weight * proto_aug_loss
            loss += args.feat_distill_weight * feat_distill_loss
            loss += args.sop_weight * sop_loss  # 添加SOP损失

            # logs
            pstr = ''
            pstr += f'me_max_loss_old_new: {me_max_loss_old_new.item():.4f} '
            pstr += f'me_max_loss_old_in: {me_max_loss_old_in.item():.4f} '
            pstr += f'me_max_loss_new_in: {me_max_loss_new_in.item():.4f} '
            pstr += f'cluster_loss: {cluster_loss.item():.4f} '
            pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '
            pstr += f'proto_aug_loss: {proto_aug_loss.item():.4f} '
            pstr += f'feat_distill_loss: {feat_distill_loss.item():.4f} '
            pstr += f'sop_loss: {sop_loss.item():.4f} '  # 记录SOP损失

            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                            .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))
                new_true_ratio = len(class_labels[class_labels>=args.num_seen_classes]) / len(class_labels)
                logits = student_out / 0.1
                preds = logits.argmax(1)
                new_pred_ratio = len(preds[preds>=args.num_seen_classes]) / len(preds)
                args.logger.info(f'Avg old prob: {torch.sum(avg_probs_old_in).item():.4f} | Avg new prob: {torch.sum(avg_probs_new_in).item():.4f} | Pred new ratio: {new_pred_ratio:.4f} | Ground-truth new ratio: {new_true_ratio:.4f}')

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        args.logger.info('Testing on disjoint test set...')
        all_acc_test, old_acc_test, new_acc_test, \
            all_acc_soft_test, seen_acc_test, unseen_acc_test = test_online(student, test_loader, epoch=epoch, save_name='Test ACC', args=args)
        args.logger.info('Test Accuracies (Hard): All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))
        args.logger.info('Test Accuracies (Soft): All {:.4f} | Seen {:.4f} | Unseen {:.4f}'.format(all_acc_soft_test, seen_acc_test, unseen_acc_test))

        # Step schedule
        exp_lr_scheduler.step()

        save_dict = {
            'model': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }

        #torch.save(save_dict, args.model_path[:-3] + '_session-' + str(current_session) + f'.pt')   # NOTE!!! session
        #args.logger.info("model saved to {}.".format(args.model_path[:-3] + '_session-' + str(current_session) + f'.pt'))

        if all_acc_test > best_test_acc_all:

            args.logger.info(f'Best ACC on All Classes on test set of session-{current_session}: {all_acc_test:.4f}...')

            torch.save(save_dict, args.model_path[:-3] + '_session-' + str(current_session) + f'_best.pt')   # NOTE!!! session
            args.logger.info("model saved to {}.".format(args.model_path[:-3] + '_session-' + str(current_session) + f'_best.pt'))

            best_test_acc_all = all_acc_test
            best_test_acc_old = old_acc_test
            best_test_acc_new = new_acc_test

            best_test_acc_soft_all = all_acc_soft_test
            best_test_acc_seen = seen_acc_test
            best_test_acc_unseen = unseen_acc_test

        args.logger.info(f'Exp Name: {args.exp_name}')
        args.logger.info(f'Metrics with best model on test set (Hard) of session-{current_session}: All (Hard): {best_test_acc_all:.4f} Old: {best_test_acc_old:.4f} New: {best_test_acc_new:.4f}')
        args.logger.info(f'Metrics with best model on test set (Hard) of session-{current_session}: All (Soft): {best_test_acc_soft_all:.4f} Seen: {best_test_acc_seen:.4f} Unseen: {best_test_acc_unseen:.4f}')
        args.logger.info('\n')


    # log best test acc list
    args.best_test_acc_all_list.append(best_test_acc_all)
    args.best_test_acc_old_list.append(best_test_acc_old)
    args.best_test_acc_new_list.append(best_test_acc_new)
    args.best_test_acc_soft_all_list.append(best_test_acc_soft_all)
    args.best_test_acc_seen_list.append(best_test_acc_seen)
    args.best_test_acc_unseen_list.append(best_test_acc_unseen)



def test_online(model, test_loader, epoch, save_name, args):

    model.eval()

    preds, targets = [], []
    mask_hard = np.array([])
    mask_soft = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, logits = model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask_hard = np.append(mask_hard, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))
            mask_soft = np.append(mask_soft, np.array([True if x.item() in range(args.num_seen_classes)
                                         else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask_hard,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    all_acc_soft, seen_acc, unseen_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask_soft,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    return all_acc, old_acc, new_acc, all_acc_soft, seen_acc, unseen_acc
'''====================================================================================================================='''


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--num_workers_test', default=4, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default='v2')

    parser.add_argument('--dataset_name', type=str, default='cifar100', help='options: cifar10, cifar100, tiny_imagenet, cub, imagenet_100')
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--exp_root', type=str, default=exp_root_happy)
    parser.add_argument('--transform', type=str, default='imagenet')

    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--contrast_unlabel_only', action='store_true', default=False)

    '''group-wise entropy regularization'''
    # memax weight for offline session
    parser.add_argument('--memax_weight', type=float, default=1)
    # memax weight for online session
    parser.add_argument('--memax_old_new_weight', type=float, default=2)
    parser.add_argument('--memax_old_in_weight', type=float, default=1)
    parser.add_argument('--memax_new_in_weight', type=float, default=1)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup) of the teacher temperature.')
    #parser.add_argument('--teacher_temp_final', default=0.05, type=float, help='Final value (online session) of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

    '''clustering-guided initialization'''
    parser.add_argument('--init_new_head', action='store_true', default=False)

    '''PASS params'''
    parser.add_argument('--proto_aug_weight', type=float, default=1.0)
    parser.add_argument('--feat_distill_weight', type=float, default=1.0)
    parser.add_argument('--radius_scale', type=float, default=1.0)

    '''hardness-aware sampling temperature'''
    parser.add_argument('--hardness_temp', type=float, default=0.1)

    '''SOP params - Selective Old-Class Protection'''
    parser.add_argument('--sop_weight', type=float, default=0.0, help='Weight for SOP loss (0=disabled)')
    parser.add_argument('--sop_lambda_max', type=float, default=1.0, help='Max regularization strength for stable old classes')
    parser.add_argument('--sop_lambda_min', type=float, default=0.1, help='Min regularization strength for unstable old classes')
    parser.add_argument('--sop_momentum', type=float, default=0.9, help='Momentum for stability score update')
    parser.add_argument('--sop_use_entropy', action='store_true', default=True, help='Use entropy as stability metric')
    parser.add_argument('--sop_update_freq', type=int, default=5, help='Frequency (epochs) to update stability scores')
    parser.add_argument('--sop_log_stability', action='store_true', default=True, help='Log stability scores and lambda weights')

    # Continual GCD params
    parser.add_argument('--num_old_classes', type=int, default=-1)
    parser.add_argument('--prop_train_labels', type=float, default=0.8)
    parser.add_argument('--train_session', type=str, default='offline', help='options: offline, online')
    parser.add_argument('--load_offline_id', type=str, default=None)
    parser.add_argument('--epochs_offline', default=100, type=int)
    parser.add_argument('--epochs_online_per_session', default=30, type=int)
    parser.add_argument('--continual_session_num', default=4, type=int)
    parser.add_argument('--online_novel_unseen_num', default=400, type=int)
    parser.add_argument('--online_old_seen_num', default=50, type=int)
    parser.add_argument('--online_novel_seen_num', default=50, type=int)

    # shuffle dataset classes
    parser.add_argument('--shuffle_classes', action='store_true', default=False)
    parser.add_argument('--seed', default=0, type=int)

    # others
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default='simgcd-pro-v5', type=str)
    parser.set_defaults(
    dataset_name='cub',
    batch_size=128,
    transform='imagenet',
    warmup_teacher_temp=0.05,
    teacher_temp=0.05,
    warmup_teacher_temp_epochs=10,
    lr=0.1,
    memax_old_new_weight=1,
    memax_old_in_weight=1,
    memax_new_in_weight=1,
    proto_aug_weight=1.0,
    feat_distill_weight=1.0,
    radius_scale=1.0,
    eval_funcs=['v2'],
    num_old_classes=100,
    prop_train_labels=0.8,
    train_session='online',
    epochs_online_per_session=20,
    continual_session_num=5,
    online_novel_unseen_num=25,
    online_old_seen_num=5,
    online_novel_seen_num=5,
    init_new_head=True,
    load_offline_id='Old100_Ratio0.8_20251111-203927',
    shuffle_classes=True,
    seed=1001
)


    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    set_seed(args.seed)
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    args.exp_root = args.exp_root + '_' + args.train_session
    args.exp_name = 'happy' + '-' + args.train_session

    if args.train_session == 'offline':
        args.base_exp_id = 'Old' + str(args.num_labeled_classes) + '_' + 'Ratio' + str(args.prop_train_labels)

    elif args.train_session == 'online':
        args.base_exp_id = 'Old' + str(args.num_labeled_classes) + '_' + 'Ratio' + str(args.prop_train_labels) \
            + '_' + 'ContinualNum' + str(args.continual_session_num) + '_' + 'UnseenNum' + str(args.online_novel_unseen_num) \
                + '_' + 'SeenNum' + str(args.online_novel_seen_num)

    else:
        raise NotImplementedError

    init_experiment(args, runner_name=['Happy'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875

    backbone = vits.__dict__['vit_base']()

    args.logger.info(f'Loading weights from {dino_pretrain_path}')
    state_dict = torch.load(dino_pretrain_path, map_location='cpu')
    backbone.load_state_dict(state_dict)

    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes   # NOTE!!!

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in backbone.parameters():
        m.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in backbone.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True

    args.logger.info('model build')

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    model = nn.Sequential(backbone, projector)

    model.to(device)

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)


    # ----------------------
    # 1. OFFLINE TRAIN
    # ----------------------
    if args.train_session == 'offline':
        args.logger.info('========== offline training with labeled old data (old) ==========')
        args.logger.info('loading dataset...')
        offline_session_train_dataset, offline_session_test_dataset,\
            _online_session_train_dataset_list, _online_session_test_dataset_list,\
                datasets, dataset_split_config_dict, novel_targets_shuffle = get_datasets(
                    args.dataset_name, train_transform, test_transform, args)

        # saving dataset dict
        print('save dataset dict...')
        save_dataset_dict_path = os.path.join(args.log_dir, 'offline_dataset_dict.txt')
        f_dataset_dict = open(save_dataset_dict_path, 'w')
        f_dataset_dict.write('offline_dataset_split_dict: \n')
        f_dataset_dict.write(str(dataset_split_config_dict))
        f_dataset_dict.write('\nnovel_targets_shuffle: \n')
        f_dataset_dict.write(str(novel_targets_shuffle))
        f_dataset_dict.close()
        
        offline_session_train_loader = DataLoader(offline_session_train_dataset, num_workers=args.num_workers,
                                                  batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)
        offline_session_test_loader = DataLoader(offline_session_test_dataset, num_workers=args.num_workers_test,
                                                 batch_size=256, shuffle=False, pin_memory=False)

        # ----------------------
        # TRAIN
        # ----------------------
        train_offline(model, offline_session_train_loader, offline_session_test_loader, args)


    # ----------------------
    # 2. ONLINE TRAIN
    # ----------------------
    elif args.train_session == 'online':
        args.logger.info('\n\n==================== online continual GCD with unlabeled data (old + novel) ====================')
        args.logger.info('loading dataset...')
        _offline_session_train_dataset, _offline_session_test_dataset,\
            online_session_train_dataset_list, online_session_test_dataset_list,\
                datasets, dataset_split_config_dict, novel_targets_shuffle = get_datasets(
                    args.dataset_name, train_transform, test_transform, args)

        # saving dataset dict
        print('save dataset dict...')
        save_dataset_dict_path = os.path.join(args.log_dir, 'online_dataset_dict.txt')
        f_dataset_dict = open(save_dataset_dict_path, 'w')
        f_dataset_dict.write('online_dataset_split_dict: \n')
        f_dataset_dict.write(str(dataset_split_config_dict))
        f_dataset_dict.write('\nnovel_targets_shuffle: \n')
        f_dataset_dict.write(str(novel_targets_shuffle))
        f_dataset_dict.write('\nnum_novel_class_per_session: \n')
        f_dataset_dict.write(str(args.num_unlabeled_classes // args.continual_session_num))
        f_dataset_dict.close()


        # ----------------------
        # CONTINUAL SESSIONS
        # ----------------------
        args.num_novel_class_per_session = args.num_unlabeled_classes // args.continual_session_num
        args.logger.info('number of novel class per session: {}'.format(args.num_novel_class_per_session))

        '''v5: ProtoAug Manager'''
        proto_aug_manager = ProtoAugManager(args.feat_dim, args.n_views*args.batch_size, args.hardness_temp, args.radius_scale, device, args.logger)

        # best test acc list across continual sessions
        args.best_test_acc_all_list = []
        args.best_test_acc_old_list = []
        args.best_test_acc_new_list = []
        args.best_test_acc_soft_all_list = []
        args.best_test_acc_seen_list = []
        args.best_test_acc_unseen_list = []

        start_session = 0

        '''Continual GCD sessions'''
        #for session in range(args.continual_session_num):
        for session in range(start_session, args.continual_session_num):
            args.logger.info('\n\n========== begin online continual session-{} ==============='.format(session+1))
            # dataset for the current session
            online_session_train_dataset = online_session_train_dataset_list[session]
            online_session_test_dataset = online_session_test_dataset_list[session]

            online_session_train_loader = DataLoader(online_session_train_dataset, num_workers=args.num_workers,
                                                     batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)
            online_session_test_loader = DataLoader(online_session_test_dataset, num_workers=args.num_workers_test,
                                                    batch_size=256, shuffle=False, pin_memory=False)

            # number of seen (offline old + previous online new) classes till the beginning of this session
            args.num_seen_classes = args.num_labeled_classes + args.num_novel_class_per_session * session
            args.logger.info('number of seen class (old + seen novel) at the beginning of current session: {}'.format(args.num_seen_classes))
            if args.dataset_name == 'cifar100':
                args.num_cur_novel_classes = len(np.unique(online_session_train_dataset.novel_unlabelled_dataset.targets))
            elif args.dataset_name == 'tiny_imagenet':
                novel_cls_labels = [t for i, (p, t) in enumerate(online_session_train_dataset.novel_unlabelled_dataset.data)]
                args.num_cur_novel_classes = len(np.unique(novel_cls_labels))
            elif args.dataset_name == 'aircraft':
                novel_cls_labels = [t for i, (p, t) in enumerate(online_session_train_dataset.novel_unlabelled_dataset.samples)]
                args.num_cur_novel_classes = len(np.unique(novel_cls_labels))
            elif args.dataset_name == 'scars':
                args.num_cur_novel_classes = len(np.unique(online_session_train_dataset.novel_unlabelled_dataset.target))   # NOTE!!! target
            else:
                args.num_cur_novel_classes = args.num_novel_class_per_session * (session+1)
            args.logger.info('number of all novel class (seen novel + unseen novel) in current session: {}'.format(args.num_cur_novel_classes))


            '''tunable params in backbone'''
            ####################################################################################################################
            # freeze backbone params
            for m in backbone.parameters():
                m.requires_grad = False

            # Only finetune layers from block 'args.grad_from_block' onwards
            for name, m in backbone.named_parameters():
                if 'block' in name:
                    block_num = int(name.split('.')[1])
                    if block_num >= args.grad_from_block:
                        m.requires_grad = True
            ####################################################################################################################

            '''load ckpts from last session (session>0) or offline session (session=0)'''
            ####################################################################################################################
            args.logger.info('loading checkpoints of model_pre...')
            if session == 0:
                projector_pre = DINOHead(in_dim=args.feat_dim, out_dim=args.num_labeled_classes, nlayers=args.num_mlp_layers)
                model_pre = nn.Sequential(backbone, projector_pre)
                if args.load_offline_id is not None:
                    load_dir_online = os.path.join(exp_root_happy + '_' + 'offline', args.dataset_name, args.load_offline_id, 'checkpoints', 'model_best.pt')
                    args.logger.info('loading offline checkpoints from: ' + load_dir_online)
                    load_dict = torch.load(load_dir_online)
                    model_pre.load_state_dict(load_dict['model'])
                    args.logger.info('successfully loaded checkpoints!')
            else:        # session > 0:
                projector_pre = DINOHead(in_dim=args.feat_dim, out_dim=args.num_seen_classes, nlayers=args.num_mlp_layers)
                model_pre = nn.Sequential(backbone, projector_pre)
                load_dir_online = args.model_path[:-3] + '_session-' + str(session) + f'_best.pt'   # NOTE!!! session, best
                args.logger.info('loading checkpoints from last online session: ' + load_dir_online)
                load_dict = torch.load(load_dir_online)
                model_pre.load_state_dict(load_dict['model'])
                args.logger.info('successfully loaded checkpoints!')
            ####################################################################################################################

            '''incremental parametric classifier in SimGCD'''
            ####################################################################################################################
            ####################################################################################################################
            backbone_cur = deepcopy(backbone)   # NOTE!!!
            backbone_cur.load_state_dict(model_pre[0].state_dict())   # NOTE!!!
            args.mlp_out_dim_cur = args.num_labeled_classes + args.num_cur_novel_classes   # total num of classes in the current session
            args.logger.info('number of all class (old + all new) in current session: {}'.format(args.mlp_out_dim_cur))
            projector_cur = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim_cur, nlayers=args.num_mlp_layers)
            args.logger.info('transferring classification head of seen classes...')
            projector_cur.last_layer.weight_v.data[:args.num_seen_classes] = projector_pre.last_layer.weight_v.data[:args.num_seen_classes]   # NOTE!!!
            projector_cur.last_layer.weight_g.data[:args.num_seen_classes] = projector_pre.last_layer.weight_g.data[:args.num_seen_classes]   # NOTE!!!
            projector_cur.last_layer.weight.data[:args.num_seen_classes] = projector_pre.last_layer.weight.data[:args.num_seen_classes]   # NOTE!!!
            # initialize new class heads
            #############################################
            online_session_train_dataset_for_new_head_init = deepcopy(online_session_train_dataset)
            online_session_train_dataset_for_new_head_init.old_unlabelled_dataset.transform = test_transform   # NOTE!!!
            online_session_train_dataset_for_new_head_init.novel_unlabelled_dataset.transform = test_transform   # NOTE!!!
            online_session_train_loader_for_new_head_init = DataLoader(online_session_train_dataset_for_new_head_init, num_workers=args.num_workers_test,
                                                                    batch_size=256, shuffle=False, pin_memory=False)
            if args.init_new_head:
                new_head = get_kmeans_centroid_for_new_head(model_pre, online_session_train_loader_for_new_head_init, args, device)   # torch.Size([10, 768])
                norm_new_head_weight_v = torch.norm(projector_cur.last_layer.weight_v.data[args.num_seen_classes:], dim=-1).mean()
                norm_new_head_weight = torch.norm(projector_cur.last_layer.weight.data[args.num_seen_classes:], dim=-1).mean()
                new_head_weight_v = new_head * norm_new_head_weight_v
                new_head_weight = new_head * norm_new_head_weight
                args.logger.info('initializing classification head of unseen novel classes...')
                projector_cur.last_layer.weight_v.data[args.num_seen_classes:] = new_head_weight_v.data   # NOTE!!!   # copy
                projector_cur.last_layer.weight.data[args.num_seen_classes:] = new_head_weight.data   # NOTE!!!
            ##############################################

            model_cur = nn.Sequential(backbone_cur, projector_cur)   # NOTE!!! backbone_cur
            args.logger.info('incremental classifier heads from {} to {}'.format(len(model_pre[1].last_layer.weight_v), len(model_cur[1].last_layer.weight_v)))
            model_cur.to(device)
            ####################################################################################################################
            ####################################################################################################################

            '''Initialize SOP module'''
            ####################################################################################################################
            if session == 0:
                # 第一个在线session,初始化SOP模块
                sop_module = SelectiveOldClassProtection(
                    num_old_classes=args.num_labeled_classes,
                    lambda_max=args.sop_lambda_max,
                    lambda_min=args.sop_lambda_min,
                    stability_momentum=args.sop_momentum,
                    use_entropy=args.sop_use_entropy
                )
                if args.sop_weight > 0:
                    args.logger.info('Initialized SOP module with {} old classes'.format(args.num_labeled_classes))
                    args.logger.info(f'SOP params: weight={args.sop_weight}, lambda_max={args.sop_lambda_max}, lambda_min={args.sop_lambda_min}')
            else:
                # 后续session,更新旧类数量为所有已见过的类
                sop_module = SelectiveOldClassProtection(
                    num_old_classes=args.num_seen_classes,
                    lambda_max=args.sop_lambda_max,
                    lambda_min=args.sop_lambda_min,
                    stability_momentum=args.sop_momentum,
                    use_entropy=args.sop_use_entropy
                )
                if args.sop_weight > 0:
                    args.logger.info('Updated SOP module with {} seen classes'.format(args.num_seen_classes))
            ####################################################################################################################

            '''compute prototypes offline (session = 0)'''
            if session == 0:
                args.logger.info('Before Train: compute offline prototypes and radius from {} classes with the best model...'.format(args.num_labeled_classes))
                offline_session_train_dataset_for_proto_aug = deepcopy(_offline_session_train_dataset)
                offline_session_train_dataset_for_proto_aug.transform = test_transform
                offline_session_train_loader_for_proto_aug = DataLoader(offline_session_train_dataset_for_proto_aug, num_workers=args.num_workers_test,
                                                                        batch_size=256, shuffle=False, pin_memory=False)
                # NOTE!!! use model_pre && offline_session_train_loader
                proto_aug_manager.update_prototypes_offline(model_pre, offline_session_train_loader_for_proto_aug, args.num_labeled_classes)
                save_path = os.path.join(args.model_dir, 'ProtoAugDict' + '_offline' + f'.pt')
                args.logger.info('Saving ProtoAugDict to {}.'.format(save_path))
                proto_aug_manager.save_proto_aug_dict(save_path)

            # ----------------------
            # TRAIN WITH SOP
            # ----------------------
            train_online(model_cur, model_pre, proto_aug_manager, sop_module, online_session_train_loader, online_session_test_loader, session+1, args)

            '''compute prototypes online after train (session > 0)'''
            #############################################################################################################
            args.logger.info('After Train: update online prototypes from {} to {} classes with the best model...'.format(args.num_seen_classes, args.num_labeled_classes + args.num_cur_novel_classes))
            # NOTE!!! use model_cur_best && online_session_train_loader
            load_dir_online_best = args.model_path[:-3] + '_session-' + str(session+1) + f'_best.pt'   # NOTE!!! session, best
            args.logger.info('loading best checkpoints current online session: ' + load_dir_online_best)
            load_dict = torch.load(load_dir_online_best)
            model_cur.load_state_dict(load_dict['model'])
            proto_aug_manager.update_prototypes_online(model_cur, online_session_train_loader_for_new_head_init, 
                                                       args.num_seen_classes, args.num_labeled_classes + args.num_cur_novel_classes)
            save_path = os.path.join(args.model_dir, 'ProtoAugDict' + '_session-' + str(session+1) + f'.pt')
            args.logger.info('Saving ProtoAugDict to {}.'.format(save_path))
            proto_aug_manager.save_proto_aug_dict(save_path)

            '''save results dict after each session'''
            best_acc_list_dict = {
                'best_test_acc_all_list': args.best_test_acc_all_list,
                'best_test_acc_old_list': args.best_test_acc_old_list,
                'best_test_acc_new_list': args.best_test_acc_new_list,
                'best_test_acc_soft_all_list': args.best_test_acc_soft_all_list,
                'best_test_acc_seen_list': args.best_test_acc_seen_list,
                'best_test_acc_unseen_list': args.best_test_acc_unseen_list,
            }
            save_results_path = os.path.join(args.model_dir, 'best_acc_list' + '_session-' + str(session+1) + f'.pt')
            args.logger.info('Saving results (best acc list) to {}.'.format(save_results_path))
            torch.save(best_acc_list_dict, save_results_path)

        # print final results
        args.logger.info('\n\n==================== print final results over {} continual sessions ===================='.format(args.continual_session_num))
        for session in range(args.continual_session_num):
            args.logger.info(f'Session-{session+1}: All (Hard): {args.best_test_acc_all_list[session]:.4f} Old: {args.best_test_acc_old_list[session]:.4f} New: {args.best_test_acc_new_list[session]:.4f} | All (Soft): {args.best_test_acc_soft_all_list[session]:.4f} Seen: {args.best_test_acc_seen_list[session]:.4f} Unseen: {args.best_test_acc_unseen_list[session]:.4f}')
        for session in range(args.continual_session_num):
            print(f'Session-{session+1}: All (Hard): {args.best_test_acc_all_list[session]:.4f} Old: {args.best_test_acc_old_list[session]:.4f} New: {args.best_test_acc_new_list[session]:.4f} | All (Soft): {args.best_test_acc_soft_all_list[session]:.4f} Seen: {args.best_test_acc_seen_list[session]:.4f} Unseen: {args.best_test_acc_unseen_list[session]:.4f}')

    else:
        raise NotImplementedError
