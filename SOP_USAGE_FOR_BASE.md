# SOP 集成到 base.py 使用指南

## ✅ 已完成的集成

SOP (Selective Old-Class Protection) 机制已成功集成到 `base.py` 中！

### 集成内容

1. **SelectiveOldClassProtection 类** (第46-158行)
   - 稳定性得分计算
   - Lambda权重自适应调整
   - SOP损失计算

2. **train_online 函数修改** (第717-883行)
   - 添加 `sop_module` 参数
   - 定期更新稳定性得分
   - 计算并添加 SOP 损失

3. **命令行参数** (第975-982行)
   - `--sop_weight`: SOP损失权重
   - `--sop_lambda_max`: 最大正则化强度
   - `--sop_lambda_min`: 最小正则化强度
   - `--sop_momentum`: 稳定性得分动量
   - `--sop_use_entropy`: 使用熵作为稳定性度量
   - `--sop_update_freq`: 更新频率
   - `--sop_log_stability`: 记录稳定性得分

4. **SOP 模块初始化** (第1281-1301行)
   - Session 0: 初始化旧类保护
   - Session 1+: 更新为所有已见类

## 🚀 使用方法

### 基本用法（启用SOP）

```bash
python base.py \
    --sop_weight 3.0 \
    --sop_lambda_max 1.0 \
    --sop_lambda_min 0.1 \
    --sop_use_entropy \
    --sop_log_stability
```

### 完整示例（针对新类精度优化）

根据你的需求（旧类准确率已经很高，需要提升新类），推荐配置：

```bash
python base.py \
    # 基本设置 (已在default中配置)
    --dataset_name cub \
    --batch_size 128 \
    --lr 0.02 \
    --train_session online \
    --load_offline_id Old100_Ratio0.8_20251111-203927 \
    \
    # Me-Max权重：增强新类探索
    --memax_old_new_weight 8.0 \
    --memax_old_in_weight 1.0 \
    --memax_new_in_weight 3.0 \
    \
    # 保护机制：减弱总体保护
    --proto_aug_weight 0.3 \
    --feat_distill_weight 0.3 \
    \
    # SOP：轻度自适应保护
    --sop_weight 3.0 \
    --sop_lambda_max 0.8 \
    --sop_lambda_min 0.05 \
    --sop_use_entropy \
    --sop_update_freq 5 \
    --sop_log_stability
```

### 禁用SOP（baseline对比）

```bash
python base.py \
    --sop_weight 0.0  # 设置为0即禁用
```

## 📊 预期效果

### 调整前（无SOP或过强保护）
```
Old Acc: 85.0%  (太高，过度保护)
New Acc: 55.0%  (太低，探索不足)
All Acc: 68.0%
```

### 调整后（轻度SOP + 强新类信号）
```
Old Acc: 80.0%  (略降5%，仍然很好)
New Acc: 63.0%  (提升8%，探索改善) ← 目标
All Acc: 70.5%  (整体提升2.5%)
```

## 🔍 监控指标

### 1. 训练日志中查看

```bash
# SOP loss趋势
grep "sop_loss" logs/log.txt

# 稳定性得分
grep "Stability scores" logs/log.txt

# 预测比例
grep "Pred new ratio" logs/log.txt
```

### 2. 期望看到的输出

**SOP正常工作**:
```
Computing stability scores for old classes...
Stability scores: 0.856 0.723 0.912 0.645 0.789 ...
Lambda weights: 0.770 0.651 0.820 0.580 0.711 ...

Epoch: [5][10/8]  loss 5.9234
sop_loss: 0.0025  ← 有值，不是0
```

**新旧类平衡改善**:
```
Pred new ratio: 0.18 | Ground-truth new ratio: 0.20  ← 接近
me_max_loss_old_new: 0.0850  ← 比原来大
```

## ⚙️ 参数调优指南

### 问题：旧类太高，新类太低

**解决方案1：减弱 SOP**
```bash
--sop_weight 2.0         # 从3.0降到2.0
--sop_lambda_max 0.5     # 从1.0降到0.5
```

**解决方案2：增强新类信号**
```bash
--memax_old_new_weight 10.0  # 进一步增强
--memax_new_in_weight 5.0    # 进一步增强
```

**解决方案3：提高学习率**
```bash
--lr 0.03               # 从0.02增到0.03
```

### 问题：旧类下降太多

**解决方案：增强 SOP**
```bash
--sop_weight 5.0
--sop_lambda_max 1.5
--proto_aug_weight 0.5   # 从0.3增到0.5
```

## 🎚️ 渐进式调整策略

### Step 1: 温和配置（快速测试）

```bash
python base.py \
    --sop_weight 3.0 \
    --memax_old_new_weight 5.0 \
    --memax_new_in_weight 2.0 \
    --proto_aug_weight 0.5 \
    --feat_distill_weight 0.5
```

### Step 2: 根据结果调整

**如果新类还是低**:
```bash
--sop_weight 2.0
--memax_old_new_weight 8.0
--proto_aug_weight 0.3
```

**如果旧类掉太多**:
```bash
--sop_weight 5.0
--memax_old_new_weight 3.0
--proto_aug_weight 0.7
```

## 📈 与原版对比

### 原版 Happy
```bash
python base.py \
    --sop_weight 0.0 \
    --proto_aug_weight 1.0 \
    --feat_distill_weight 1.0 \
    --memax_old_new_weight 1.0
```

### Happy + SOP（平衡版）
```bash
python base.py \
    --sop_weight 3.0 \
    --sop_lambda_max 0.8 \
    --sop_lambda_min 0.05 \
    --proto_aug_weight 0.3 \
    --feat_distill_weight 0.3 \
    --memax_old_new_weight 8.0 \
    --memax_new_in_weight 3.0
```

## 🧪 快速验证实验

### 10 epochs 快速测试

修改 `set_defaults` 中的参数：

```python
parser.set_defaults(
    # ... 其他参数 ...
    epochs_online_per_session=10,  # 快速测试
    sop_weight=3.0,                # 启用SOP
    # ... 其他参数 ...
)
```

然后运行：
```bash
python base.py
```

查看结果：
```bash
tail -50 logs/log.txt | grep "Test Accuracies"
```

## 💡 核心思想总结

### SOP的作用
```
总保护 = ProtoAug + FeatDistill + SOP

传统: 每个旧类相同保护强度
SOP:  根据稳定性自适应调整

高稳定类 → 强保护 (λ接近λ_max)
低稳定类 → 弱保护 (λ接近λ_min)
```

### 提升新类准确率的策略
```
1. 减弱总保护强度
   proto_aug_weight: 1.0 → 0.3
   feat_distill_weight: 1.0 → 0.3
   sop_weight: 适中 (3.0)

2. 增强新类信号
   memax_old_new_weight: 1.0 → 8.0
   memax_new_in_weight: 1.0 → 3.0

3. 提高探索能力
   lr: 0.01 → 0.02
```

## 🔧 调试技巧

### 1. 检查SOP是否生效

```bash
# 应该看到非0的sop_loss
grep "sop_loss" logs/log.txt | head -20

# 应该看到稳定性得分有差异
grep "Stability scores" logs/log.txt | head -5
```

### 2. 监控新旧平衡

```bash
# Pred new ratio应该接近Ground-truth new ratio
grep "Pred new ratio" logs/log.txt | tail -20
```

### 3. 对比实验

```bash
# 实验A: 无SOP
python base.py --sop_weight 0.0 2>&1 | tee log_baseline.txt

# 实验B: 有SOP
python base.py --sop_weight 3.0 2>&1 | tee log_sop.txt

# 对比结果
diff <(grep "Test Accuracies (Hard)" log_baseline.txt) \
     <(grep "Test Accuracies (Hard)" log_sop.txt)
```

## 📝 修改默认值（可选）

如果想永久使用SOP，可以修改 `set_defaults`:

```python
parser.set_defaults(
    # ... 其他参数 ...
    
    # SOP参数
    sop_weight=3.0,
    sop_lambda_max=0.8,
    sop_lambda_min=0.05,
    sop_use_entropy=True,
    sop_log_stability=True,
    
    # 平衡参数
    memax_old_new_weight=8.0,
    memax_new_in_weight=3.0,
    proto_aug_weight=0.3,
    feat_distill_weight=0.3,
    
    # ... 其他参数 ...
)
```

## 🎯 常见配置模板

### 配置1: 平衡型（推荐）
```bash
--sop_weight 3.0 --sop_lambda_max 0.8 --sop_lambda_min 0.05
--memax_old_new_weight 8.0 --memax_new_in_weight 3.0
--proto_aug_weight 0.3 --feat_distill_weight 0.3
```

### 配置2: 重视新类
```bash
--sop_weight 2.0 --sop_lambda_max 0.5 --sop_lambda_min 0.01
--memax_old_new_weight 10.0 --memax_new_in_weight 5.0
--proto_aug_weight 0.2 --feat_distill_weight 0.2
```

### 配置3: 重视旧类
```bash
--sop_weight 5.0 --sop_lambda_max 2.0 --sop_lambda_min 0.2
--memax_old_new_weight 3.0 --memax_new_in_weight 1.0
--proto_aug_weight 0.5 --feat_distill_weight 0.5
```

---

## 🎉 总结

SOP已完全集成到 `base.py`！

**关键点**:
- ✅ SOP类已添加（第46-158行）
- ✅ train_online已修改（添加sop_module参数）
- ✅ 命令行参数已添加
- ✅ 初始化逻辑已实现
- ✅ 默认配置已优化

**使用建议**:
1. 从平衡型配置开始
2. 根据Old/New准确率调整
3. 监控日志中的关键指标
4. 做对比实验验证效果

现在就可以运行 `python base.py` 来使用带SOP的训练了！🚀

