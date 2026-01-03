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

# 路径定义
dino_pretrain_path = '/home/asus/GPC-master/checkpoints/dino_vitbase16_pretrain.pth'
exp_root_happy = '/home/asus/Happy-CGCD3/dev_outputs_Happy'  

@torch.no_grad()
def sinkhorn_knopp(logits, iterations=3, epsilon=0.05):
    """
    DOTR (方案 3) 核心逻辑：最优传输算法。
    强制 Batch 内样本在所有类别上的预测分布趋于均衡。
    """
    Q = torch.exp(logits / epsilon).t() # [K, B]
    B, K = Q.shape[1], Q.shape[0]
    Q /= Q.sum()
    for _ in range(iterations):
        Q /= Q.sum(dim=1, keepdim=True); Q /= K
        Q /= Q.sum(dim=0, keepdim=True); Q /= B
    return (Q * B).t() # [B, K]

class ProtoAugManager:
    def __init__(self, feature_dim, batch_size, hardness_temp, radius_scale, device, logger):
        self.feature_dim, self.batch_size, self.device = feature_dim, batch_size, device
        self.prototypes, self.mean_similarity, self.radius = None, None, 0
        self.hardness_temp, self.radius_scale, self.logger = hardness_temp, radius_scale, logger

    def save_proto_aug_dict(self, save_path):
        torch.save({'prototypes': self.prototypes, 'radius': self.radius, 'mean_similarity': self.mean_similarity}, save_path)

    def load_proto_aug_dict(self, load_path):
        d = torch.load(load_path)
        self.prototypes, self.radius, self.mean_similarity = d['prototypes'], d['radius'], d['mean_similarity']

    def compute_proto_aug_hardness_aware_loss(self, model):
        prototypes = F.normalize(self.prototypes, dim=-1, p=2).to(self.device)
        prob = F.softmax(self.mean_similarity / self.hardness_temp, dim=-1).cpu().numpy()
        labels = torch.from_numpy(np.random.choice(len(prototypes), size=(self.batch_size,), replace=True, p=prob)).long().to(self.device)
        aug = prototypes[labels] + torch.randn((self.batch_size, self.feature_dim), device=self.device) * self.radius * self.radius_scale
        _, out = model[1](aug)
        return nn.CrossEntropyLoss()(out / 0.1, labels)

    def update_prototypes_offline(self, model, train_loader, num_labeled_classes):
        model.eval(); all_f, all_l = [], []
        for img, lab, _ in tqdm(train_loader):
            with torch.no_grad():
                all_f.append(F.normalize(model[0](img.cuda()), -1)); all_l.append(lab)
        all_f, all_l = torch.cat(all_f, 0), torch.cat(all_l, 0)
        p_list, r_list = [], []
        for c in range(num_labeled_classes):
            f_c = all_f[all_l==c]; m_c = torch.mean(f_c, 0); p_list.append(m_c)
            r_list.append(torch.trace(torch.matmul((f_c-m_c).t(), (f_c-m_c))/len(f_c))/self.feature_dim)
        self.radius = torch.sqrt(torch.mean(torch.stack(r_list)))
        self.prototypes = F.normalize(torch.stack(p_list), -1)
        sim = self.prototypes @ self.prototypes.T
        for i in range(len(sim)): sim[i,i] = 0
        self.mean_similarity = torch.sum(sim, -1) / (len(sim)-1)

    def update_prototypes_online(self, model, train_loader, num_seen, num_all):
        model.eval(); all_f, all_p = [], []
        for img, _, _, _ in tqdm(train_loader):
            with torch.no_grad():
                _, logits = model(img.cuda())
                all_f.append(F.normalize(model[0](img.cuda()), -1)); all_p.append(logits.argmax(1))
        all_f, all_p = torch.cat(all_f, 0), torch.cat(all_p, 0)
        p_list = []
        for c in range(num_seen, num_all):
            f_c = all_f[all_p==c]
            p_list.append(torch.mean(f_c, 0) if len(f_c)>0 else model[1].last_layer.weight_v.data[c])
        self.prototypes = F.normalize(torch.cat([self.prototypes, torch.stack(p_list)], 0), -1)
        sim = self.prototypes @ self.prototypes.T
        for i in range(len(sim)): sim[i,i] = 0
        self.mean_similarity = torch.sum(sim, -1) / (len(sim)-1)

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim)]
        if use_bn: layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_bn: layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer: self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_proj = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        logits = self.last_layer(x)
        return x_proj, logits

def get_params_groups(model):
    regularized, not_regularized = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if name.endswith(".bias") or len(param.shape) == 1: not_regularized.append(param)
        else: regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super().__init__()
        self.temperature, self.contrast_mode, self.base_temperature = temperature, contrast_mode, base_temperature

    def forward(self, features, labels=None, mask=None):
        if len(features.shape) > 3: features = features.view(features.shape[0], features.shape[1], -1)
        batch_size = features.shape[0]
        if labels is not None:
            mask = torch.eq(labels.view(-1, 1), labels.view(-1, 1).T).float().to(features.device)
        else:
            mask = torch.eye(batch_size, dtype=torch.float32).to(features.device)
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), 0)
        anchor_feature = contrast_feature if self.contrast_mode == 'all' else features[:, 0]
        anchor_count = contrast_count if self.contrast_mode == 'all' else 1
        logits = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1).to(features.device), 0)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        return - (self.temperature / self.base_temperature) * mean_log_prob_pos.view(anchor_count, batch_size).mean()

def info_nce_logits(features, n_views=2, temperature=1.0, device='cuda'):
    b_ = 0.5 * int(features.size(0))
    labels = torch.cat([torch.arange(b_) for _ in range(n_views)], 0).to(device)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    features = F.normalize(features, 1)
    sim = torch.matmul(features, features.T)
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    sim, labels = sim[~mask].view(sim.shape[0], -1), labels[~mask].view(labels.shape[0], -1)
    positives = sim[labels.bool()].view(labels.shape[0], -1)
    negatives = sim[~labels.bool()].view(sim.shape[0], -1)
    logits = torch.cat([positives, negatives], 1) / temperature
    return logits, torch.zeros(logits.shape[0], dtype=torch.long).to(device)

class DistillLoss(nn.Module):
    def __init__(self, warmup_teacher_temp_epochs, nepochs, ncrops=2, warmup_teacher_temp=0.07, teacher_temp=0.04, student_temp=0.1):
        super().__init__()
        self.student_temp, self.ncrops = student_temp, ncrops
        self.teacher_temp_schedule = np.concatenate((np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs), np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp))

    def forward(self, student_output, teacher_output, epoch):
        student_out = (student_output / self.student_temp).chunk(self.ncrops)
        teacher_out = F.softmax(teacher_output / self.teacher_temp_schedule[epoch], dim=-1).detach().chunk(2)
        total_loss, n_terms = 0, 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq: continue
                total_loss += torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1).mean(); n_terms += 1
        return total_loss / n_terms

def get_kmeans_centroid_for_new_head(model, loader, args, device):
    model.eval(); all_f = []
    with torch.no_grad():
        for img, _, _, _ in tqdm(loader): all_f.append(F.normalize(model[0](img.cuda()), -1).cpu().numpy())
    all_f = np.concatenate(all_f)
    kmeans = KMeans(n_clusters=args.num_labeled_classes+args.num_cur_novel_classes, random_state=0).fit(all_f)
    centroids = F.normalize(torch.from_numpy(kmeans.cluster_centers_).to(device), -1)
    with torch.no_grad():
        _, logits = model[1](centroids); _, idx = torch.topk(torch.max(logits, -1)[0], k=args.num_novel_class_per_session, largest=False)
    return centroids[idx]

def train_offline(student, train_loader, test_loader, args):
    opt = SGD(get_params_groups(student), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    sch = lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs_offline, eta_min=args.lr * 1e-3)
    cluster_crit = DistillLoss(args.warmup_teacher_temp_epochs, args.epochs_offline, args.n_views, args.warmup_teacher_temp, args.teacher_temp)
    best_acc = 0
    for epoch in range(args.epochs_offline):
        student.train(); loss_rec = AverageMeter()
        for batch in train_loader:
            img, lab = batch[0], batch[1].cuda(); img = torch.cat(img, 0).cuda()
            proj, out = student(img)
            cls_loss = nn.CrossEntropyLoss()(torch.cat([f for f in (out/0.1).chunk(2)],0), torch.cat([lab for _ in range(2)],0))
            cluster_loss = cluster_crit(out, out.detach(), epoch)
            avg_p = (out/0.1).softmax(1).mean(0)
            me_max_loss = -torch.sum(torch.log(avg_p**(-avg_p))) + math.log(float(len(avg_p)))
            c_logits, c_labels = info_nce_logits(proj)
            contrast_loss = nn.CrossEntropyLoss()(c_logits, c_labels)
            student_proj = torch.nn.functional.normalize(torch.cat([f.unsqueeze(1) for f in proj.chunk(2)], 1), -1)
            sup_con_loss = SupConLoss()(student_proj, labels=lab)
            loss = (1-args.sup_weight)*(cluster_loss+contrast_loss+args.memax_weight*me_max_loss) + args.sup_weight*(cls_loss+sup_con_loss)
            opt.zero_grad(); loss.backward(); opt.step(); loss_rec.update(loss.item(), lab.size(0))
        all_acc, old_acc, _ = test_offline(student, test_loader, epoch, 'Test', args)
        sch.step()
        if old_acc > best_acc:
            best_acc = old_acc; torch.save({'model': student.state_dict()}, args.model_path[:-3] + '_best.pt')

def test_offline(model, loader, epoch, name, args):
    model.eval(); p, t, m = [], [], []
    for img, lab, _ in tqdm(loader):
        with torch.no_grad():
            _, out = model(img.cuda()); p.append(out.argmax(1).cpu().numpy()); t.append(lab.cpu().numpy())
            m.append([x.item() in range(len(args.train_classes)) for x in lab])
    return log_accs_from_preds(np.concatenate(t), np.concatenate(p), np.concatenate(m), epoch, args.eval_funcs, name, args)

def train_online(student, student_pre, pa_manager, train_loader, test_loader, session, args):
    opt = SGD(get_params_groups(student), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    sch = lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs_online_per_session, eta_min=args.lr * 1e-3)
    best_acc = 0
    for epoch in range(args.epochs_online_per_session):
        student.train(); student_pre.eval(); loss_rec = AverageMeter()
        for batch in train_loader:
            img, lab = batch[0], batch[1].cuda(); img = torch.cat(img, 0).cuda()
            proj, out = student(img)
            
            # --- DOTR (方案 3) 核心改进：动态伪标签细化 ---
            with torch.no_grad():
                # 获取具备均衡分布约束的目标伪标签分布 Q
                target_q = sinkhorn_knopp(out.detach())
            
            # 软标签交叉熵损失：替代了原版的手动熵正则项
            dotr_loss = -torch.mean(torch.sum(target_q * F.log_softmax(out / 0.1, dim=1), dim=1))
            # -----------------------------------------------

            pa_loss = pa_manager.compute_proto_aug_hardness_aware_loss(student)
            feats = F.normalize(student[0](img), -1)
            with torch.no_grad(): f_pre = F.normalize(student_pre[0](img), -1)
            distill_loss = (feats - f_pre).pow(2).sum() / len(feats)

            # 总损失包含 DOTR 自监督、原型回放和特征蒸馏
            loss = dotr_loss + args.proto_aug_weight*pa_loss + args.feat_distill_weight*distill_loss
            
            opt.zero_grad(); loss.backward(); opt.step(); loss_rec.update(loss.item(), lab.size(0))
        res = test_online(student, test_loader, epoch, 'Test', args); sch.step()
        if res[0] > best_acc:
            best_acc = res[0]; torch.save({'model': student.state_dict()}, args.model_path[:-3] + f'_session-{session}_best.pt')
    args.best_test_acc_all_list.append(best_acc)

def test_online(model, loader, epoch, name, args):
    model.eval(); p, t, mh, ms = [], [], [], []
    for img, lab, _ in tqdm(loader):
        with torch.no_grad():
            _, out = model(img.cuda()); p.append(out.argmax(1).cpu().numpy()); t.append(lab.cpu().numpy())
            mh.append([x.item() in range(len(args.train_classes)) for x in lab])
            ms.append([x.item() in range(args.num_seen_classes) for x in lab])
    t, p = np.concatenate(t), np.concatenate(p)
    return log_accs_from_preds(t, p, np.concatenate(mh), epoch, args.eval_funcs, name, args) + \
           log_accs_from_preds(t, p, np.concatenate(ms), epoch, args.eval_funcs, name, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cluster'); parser.add_argument('--dataset_name', type=str, default='cub')
    parser.add_argument('--batch_size', default=128, type=int); parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--epochs_offline', default=100, type=int); parser.add_argument('--epochs_online_per_session', default=20, type=int)
    parser.add_argument('--continual_session_num', default=5, type=int); parser.add_argument('--seed', default=1001, type=int)
    parser.add_argument('--train_session', type=str, default='online'); parser.add_argument('--load_offline_id', type=str, default='')
    # 补齐参数
    args = parser.parse_args(); args.momentum, args.weight_decay, args.grad_from_block = 0.9, 5e-5, 11
    args.sup_weight, args.n_views, args.memax_weight, args.memax_old_new_weight = 0.35, 2, 1, 1
    args.warmup_teacher_temp, args.teacher_temp, args.warmup_teacher_temp_epochs = 0.05, 0.05, 10
    args.proto_aug_weight, args.feat_distill_weight, args.radius_scale, args.hardness_temp = 1.0, 1.0, 1.0, 0.1
    args.init_new_head, args.prop_train_labels, args.online_novel_unseen_num = True, 0.8, 25
    args.online_old_seen_num, args.online_novel_seen_num, args.shuffle_classes, args.transform = 5, 5, True, 'imagenet'
    args.eval_funcs, args.num_workers, args.num_workers_test = ['v2'], 4, 4; args.exp_root = exp_root_happy
    
    device = torch.device('cuda:0'); set_seed(args.seed); args = get_class_splits(args)
    args.num_labeled_classes, args.num_unlabeled_classes = len(args.train_classes), len(args.unlabeled_classes)
    args.exp_root += '_' + args.train_session; args.exp_name = 'happy-' + args.train_session
    init_experiment(args, runner_name=['Happy'])
    
    backbone = vits.__dict__['vit_base']()
    backbone.load_state_dict(torch.load(dino_pretrain_path, map_location='cpu'))
    for name, m in backbone.named_parameters():
        m.requires_grad = False
        if 'block' in name and int(name.split('.')[1]) >= args.grad_from_block: m.requires_grad = True
    
    args.image_size, args.feat_dim, args.num_mlp_layers, args.mlp_out_dim = 224, 768, 3, args.num_labeled_classes
    model = nn.Sequential(backbone, DINOHead(args.feat_dim, args.mlp_out_dim)).to(device)
    train_tr, test_tr = get_transform(args.transform, image_size=args.image_size, args=args)
    train_tr = ContrastiveLearningViewGenerator(train_tr, n_views=args.n_views)
    
    if args.train_session == 'offline':
        ds_tr, ds_te, _, _, _, _, _ = get_datasets(args.dataset_name, train_tr, test_tr, args)
        train_offline(model, DataLoader(ds_tr, args.batch_size, True, drop_last=True), DataLoader(ds_te, 256, False), args)
    elif args.train_session == 'online':
        _, _, ds_list_tr, ds_list_te, ads, _, _ = get_datasets(args.dataset_name, train_tr, test_tr, args)
        args.num_novel_class_per_session = args.num_unlabeled_classes // args.continual_session_num
        pa_manager = ProtoAugManager(args.feat_dim, args.n_views*args.batch_size, args.hardness_temp, args.radius_scale, device, args.logger)
        args.best_test_acc_all_list = []
        
        for s in range(args.continual_session_num):
            args.num_seen_classes = args.num_labeled_classes + args.num_novel_class_per_session * s
            args.num_cur_novel_classes = args.num_novel_class_per_session * (s+1)
            if s == 0:
                m_pre = nn.Sequential(deepcopy(backbone), DINOHead(args.feat_dim, args.num_labeled_classes)).to(device)
                if args.load_offline_id: m_pre.load_state_dict(torch.load(os.path.join(exp_root_happy+'_offline', args.dataset_name, args.load_offline_id, 'checkpoints', 'model_best.pt'))['model'])
            else:
                m_pre = nn.Sequential(deepcopy(backbone), DINOHead(args.feat_dim, args.num_seen_classes)).to(device)
                m_pre.load_state_dict(torch.load(args.model_path[:-3] + f'_session-{s}_best.pt')['model'])
            
            p_cur = DINOHead(args.feat_dim, args.num_labeled_classes + args.num_cur_novel_classes).to(device)
            p_cur.last_layer.weight_v.data[:args.num_seen_classes] = m_pre[1].last_layer.weight_v.data
            if args.init_new_head:
                tmp_ds = deepcopy(ds_list_tr[s]); tmp_ds.old_unlabelled_dataset.transform = test_tr; tmp_ds.novel_unlabelled_dataset.transform = test_tr
                nh = get_kmeans_centroid_for_new_head(m_pre, DataLoader(tmp_ds, 256, False), args, device)
                p_cur.last_layer.weight_v.data[args.num_seen_classes:] = nh * torch.norm(p_cur.last_layer.weight_v.data[:args.num_seen_classes], -1).mean()
            
            m_cur = nn.Sequential(deepcopy(backbone), p_cur).to(device); m_cur[0].load_state_dict(m_pre[0].state_dict())
            if s == 0:
                tmp_ads = deepcopy(ads); tmp_ads.transform = test_tr; pa_manager.update_prototypes_offline(m_pre, DataLoader(tmp_ads, 256, False), args.num_labeled_classes)
            
            train_online(m_cur, m_pre, pa_manager, DataLoader(ds_list_tr[s], args.batch_size, True, drop_last=True), DataLoader(ds_list_te[s], 256, False), s+1, args)
            m_cur.load_state_dict(torch.load(args.model_path[:-3] + f'_session-{s+1}_best.pt')['model'])
            tmp_ds = deepcopy(ds_list_tr[s]); tmp_ds.old_unlabelled_dataset.transform = test_tr; tmp_ds.novel_unlabelled_dataset.transform = test_tr
            pa_manager.update_prototypes_online(m_cur, DataLoader(tmp_ds, 256, False), args.num_seen_classes, args.num_labeled_classes + args.num_cur_novel_classes)

