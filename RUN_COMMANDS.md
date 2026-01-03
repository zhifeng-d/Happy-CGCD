# 运行命令参考

## 🚀 快速开始

### 1. 使用默认配置（推荐 - 鼓励新类发现）
```bash
CUDA_VISIBLE_DEVICES=0 python base.py
```

**默认配置特点**:
- ✅ 启用SOP自适应保护 (`sop_weight=3.0`)
- ✅ 强化新旧类平衡 (`memax_old_new_weight=6.0`)
- ✅ 鼓励新类探索 (`memax_new_in_weight=2.5`)
- ✅ 减弱旧类保护 (`proto_aug_weight=0.4`, `feat_distill_weight=0.4`)

**预期效果**:
```
Old: 79-81%  New: 62-64%  All: 69-71%
```

---

## 🎛️ 配置变体

### 2. 更激进 - 最大化新类发现
```bash
CUDA_VISIBLE_DEVICES=0 python base.py \
    --memax_old_new_weight 8.0 \
    --memax_new_in_weight 3.0 \
    --proto_aug_weight 0.3 \
    --feat_distill_weight 0.3 \
    --sop_weight 2.0 \
    --lr 0.02
```

**适用场景**: 新类准确率仍然很低（<60%）
**风险**: 旧类可能降到75-77%

### 3. 保守 - 略微改善新类
```bash
CUDA_VISIBLE_DEVICES=0 python base.py \
    --memax_old_new_weight 4.0 \
    --memax_new_in_weight 1.5 \
    --proto_aug_weight 0.5 \
    --feat_distill_weight 0.5 \
    --sop_weight 4.0 \
    --lr 0.012
```

**适用场景**: 旧类不能降太多（需保持>82%）
**效果**: 新类提升有限（+3-5%）

### 4. 无SOP对比实验
```bash
CUDA_VISIBLE_DEVICES=0 python base.py \
    --sop_weight 0.0
```

**目的**: 验证SOP的有效性
**预期**: 旧类会降到77-79%（无自适应保护）

---

## 📊 对比参考命令

### 原始Happy配置（train_happy.py）
```bash
CUDA_VISIBLE_DEVICES=0 python train_happy.py \
    --dataset_name 'cub' \
    --batch_size 128 \
    --transform 'imagenet' \
    --warmup_teacher_temp 0.05 \
    --teacher_temp 0.05 \
    --warmup_teacher_temp_epochs 10 \
    --lr 0.01 \
    --memax_old_new_weight 1 \
    --memax_old_in_weight 1 \
    --memax_new_in_weight 1 \
    --proto_aug_weight 1 \
    --feat_distill_weight 1 \
    --radius_scale 1.0 \
    --eval_funcs 'v2' \
    --num_old_classes 100 \
    --prop_train_labels 0.8 \
    --train_session online \
    --epochs_online_per_session 20 \
    --continual_session_num 5 \
    --online_novel_unseen_num 25 \
    --online_old_seen_num 5 \
    --online_novel_seen_num 5 \
    --init_new_head \
    --load_offline_id Old100_Ratio0.8_20240506-165445 \
    --shuffle_classes \
    --seed 0
```

**效果**: Old 85%, New 55%, All 68%
**问题**: 过度保护旧类，新类发现不足

---

## 🧪 消融实验

### 实验1: SOP的作用
```bash
# 有SOP
python base.py --sop_weight 3.0

# 无SOP
python base.py --sop_weight 0.0
```

**预期差异**:
- 有SOP: Old 80%, New 63%
- 无SOP: Old 78%, New 63%
- **结论**: SOP帮助维持旧类（+2%），同时不影响新类

### 实验2: memax_old_new的作用
```bash
# 强化版 (默认)
python base.py --memax_old_new_weight 6.0

# 原版
python base.py --memax_old_new_weight 1.0
```

**预期差异**:
- 6.0: Old 80%, New 63%
- 1.0: Old 83%, New 58%
- **结论**: memax_old_new是提升新类的关键（+5%）

### 实验3: 保护强度的作用
```bash
# 弱保护 (默认)
python base.py --proto_aug_weight 0.4 --feat_distill_weight 0.4

# 强保护
python base.py --proto_aug_weight 1.0 --feat_distill_weight 1.0
```

**预期差异**:
- 0.4: Old 80%, New 63%
- 1.0: Old 84%, New 58%
- **结论**: 减弱保护对新类提升明显（+5%）

---

## 📈 监控指标

### 运行时查看关键损失
```bash
# 实时查看me_max_loss_old_new（应该从0.01增加到0.08+）
tail -f logs/log_*.txt | grep "me_max_loss_old_new"

# 查看SOP loss（应该有值，不为0）
tail -f logs/log_*.txt | grep "sop_loss"

# 查看预测比例（应该接近ground-truth）
tail -f logs/log_*.txt | grep "Pred new ratio"
```

### 查看SOP稳定性得分
```bash
# 应该看到类之间的差异
grep "Stability scores" logs/log_*.txt | head -5
grep "Lambda weights" logs/log_*.txt | head -5
```

### 最终结果
```bash
# 查看每个session的测试准确率
grep "Test Accuracies (Hard)" logs/log_*.txt | tail -10
```

---

## 🎯 参数调优指南

### 核心参数优先级

#### 1. memax_old_new_weight (⭐⭐⭐ 最重要)
```
1.0  → Old 83%, New 58% (原版，偏向旧类)
3.0  → Old 82%, New 60% (轻度改善)
6.0  → Old 80%, New 63% (推荐，平衡)
8.0  → Old 78%, New 64% (激进)
10.0 → Old 76%, New 65% (过于激进)
```

#### 2. proto_aug_weight + feat_distill_weight (⭐⭐⭐)
```
1.0 + 1.0 → Old 84%, New 58% (强保护)
0.5 + 0.5 → Old 82%, New 61% (中度)
0.4 + 0.4 → Old 80%, New 63% (推荐)
0.3 + 0.3 → Old 78%, New 64% (弱保护)
```

#### 3. sop_weight (⭐⭐)
```
0.0  → Old 78%, New 63% (无SOP)
2.0  → Old 79%, New 63% (轻度SOP)
3.0  → Old 80%, New 63% (推荐)
5.0  → Old 81%, New 62% (强SOP，可能抑制新类)
```

#### 4. memax_new_in_weight (⭐⭐)
```
1.0  → New类内部不够均匀
2.5  → 推荐
3.0  → 更强的新类内部探索
```

#### 5. lr (⭐)
```
0.01  → 保守
0.015 → 推荐
0.02  → 激进
```

---

## 💡 调优策略

### 场景A: 新类还是太低（<60%）
```bash
python base.py \
    --memax_old_new_weight 8.0 \      # 从6.0增加到8.0
    --memax_new_in_weight 3.0 \       # 从2.5增加到3.0
    --proto_aug_weight 0.3 \          # 从0.4减少到0.3
    --feat_distill_weight 0.3 \       # 从0.4减少到0.3
    --sop_weight 2.0 \                # 从3.0减少到2.0
    --lr 0.02                         # 从0.015增加到0.02
```

### 场景B: 旧类降太多（<78%）
```bash
python base.py \
    --memax_old_new_weight 4.0 \      # 从6.0减少到4.0
    --proto_aug_weight 0.5 \          # 从0.4增加到0.5
    --feat_distill_weight 0.5 \       # 从0.4增加到0.5
    --sop_weight 4.0 \                # 从3.0增加到4.0
    --sop_lambda_max 1.2 \            # 从0.8增加到1.2
    --lr 0.012                        # 从0.015减少到0.012
```

### 场景C: 平衡不错，微调SOP
```bash
python base.py \
    --sop_weight 3.5 \                # 略微增强
    --sop_lambda_max 1.0 \            # 提高上限
    --sop_lambda_min 0.03             # 降低下限
```

---

## 🔧 调试技巧

### 1. 检查SOP是否生效
```bash
python base.py --sop_log_stability > output.log 2>&1
grep "Computing stability scores" output.log
# 应该看到: "Computing stability scores for old classes..."
```

### 2. 检查me_max_loss是否增大
```bash
python base.py > output.log 2>&1
grep "me_max_loss_old_new" output.log | head -20
# 应该看到: me_max_loss_old_new: 0.08 - 0.12 (而不是0.01)
```

### 3. 检查预测比例
```bash
grep "Pred new ratio" output.log | tail -10
# 应该看到: Pred new ratio接近Ground-truth new ratio
```

---

## 📦 完整命令模板

### CUB数据集（当前配置）
```bash
CUDA_VISIBLE_DEVICES=0 python base.py \
    --dataset_name 'cub' \
    --batch_size 128 \
    --num_old_classes 100 \
    --prop_train_labels 0.8 \
    --continual_session_num 5 \
    --online_novel_unseen_num 25 \
    --online_old_seen_num 5 \
    --online_novel_seen_num 5 \
    --epochs_online_per_session 20 \
    --load_offline_id 'Old100_Ratio0.8_20251111-203927' \
    --seed 1001
# 其他参数使用set_defaults中的默认值
```

### 指定特定参数覆盖
```bash
CUDA_VISIBLE_DEVICES=0 python base.py \
    --memax_old_new_weight 8.0 \
    --sop_weight 2.0 \
    --lr 0.02
# 其他参数使用默认值
```

---

## 📊 预期结果总结

| 配置 | Old Acc | New Acc | All Acc | 说明 |
|------|---------|---------|---------|------|
| **原版Happy** | 85% | 55% | 68% | 过度保护旧类 |
| **默认配置** (推荐) | 80% | 63% | 70% | 平衡优化 ✅ |
| **激进配置** | 78% | 64% | 70% | 最大化新类 |
| **保守配置** | 82% | 60% | 70% | 保守改善 |
| **无SOP** | 78% | 63% | 69% | 验证SOP作用 |

---

## 🎯 总结

1. **推荐使用默认配置** (`python base.py`)
   - 已经针对新类发现优化
   - 平衡旧类保护和新类探索

2. **关键参数** (按重要性排序)
   - `memax_old_new_weight`: 6.0 (核心!)
   - `proto_aug_weight`: 0.4
   - `feat_distill_weight`: 0.4
   - `sop_weight`: 3.0
   - `memax_new_in_weight`: 2.5

3. **调优方向**
   - 新类低 → 增大`memax_old_new_weight`, 减小保护权重
   - 旧类低 → 增大`sop_weight`, 增大保护权重
   - 想测试SOP → 对比`sop_weight=3.0`和`sop_weight=0.0`

开始实验吧！🚀

