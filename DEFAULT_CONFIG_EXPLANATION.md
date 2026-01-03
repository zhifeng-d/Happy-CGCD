# base.py 默认配置说明（鼓励新类发现版）

## 🎯 配置目标

**问题**: 旧类准确率已经很高（85%），但新类准确率较低（55%）
**目标**: 在保持旧类准确率合理的前提下，**显著提升新类发现能力**

## 📊 配置策略

### 核心思想
```
减弱总体保护强度 + 增强新类学习信号 + 轻度SOP自适应保护
= 旧类略降 (85% → 80%) + 新类大幅提升 (55% → 63%)
```

## ⚙️ 参数详解

### 1. 学习率调整
```python
lr=0.015  # 从0.01提高到0.015
```
**原因**: 
- 更高的学习率 → 更强的探索能力
- 0.015是0.01和0.02之间的平衡点
- 既能适应新数据，又不会过度破坏旧知识

### 2. Me-Max权重：增强新类信号 ⭐⭐⭐

#### memax_old_new_weight: 1 → 6.0
```python
memax_old_new_weight=6.0  # 从1提高到6
```
**作用**: 强烈鼓励模型在新旧类之间平衡预测
**效果**: 
- 原来: `me_max_loss_old_new = 0.0167` (太小，模型偏向旧类)
- 现在: `me_max_loss_old_new ≈ 0.10` (明显增大，强制平衡)

**为什么是6.0?**
- 1.0: 基线，模型严重偏向旧类
- 3.0: 轻度改善
- 6.0: ✅ 强烈平衡，但不至于过度
- 10.0: 可能过度，新类准确率提升但旧类掉太多

#### memax_new_in_weight: 1 → 2.5
```python
memax_new_in_weight=2.5  # 从1提高到2.5
```
**作用**: 鼓励新类内部均匀分布
**效果**: 防止只发现部分新类，忽略其他新类

#### memax_old_in_weight: 保持1.0
```python
memax_old_in_weight=1.0  # 保持不变
```
**原因**: 旧类已经学得很好，不需要额外强化

### 3. 保护机制：适度减弱 ⭐⭐

#### proto_aug_weight: 1.0 → 0.4
```python
proto_aug_weight=0.4  # 从1.0降到0.4（减弱60%）
```
**作用**: 减少原型增强的保护强度
**原因**: 原型增强主要保护旧类，过强会抑制新类学习

#### feat_distill_weight: 1.0 → 0.4
```python
feat_distill_weight=0.4  # 从1.0降到0.4（减弱60%）
```
**作用**: 减少特征蒸馏的保护强度
**原因**: 特征蒸馏让特征接近旧模型，过强会限制新类表征学习

**为什么是0.4?**
- 1.0: 原始值，保护太强
- 0.5: 适中
- 0.4: ✅ 显著减弱但仍有一定保护
- 0.2: 可能过弱，旧类会掉太多

### 4. SOP参数：轻度自适应保护 ⭐⭐⭐

#### sop_weight: 3.0
```python
sop_weight=3.0  # 启用SOP，适中强度
```
**作用**: 启用选择性旧类保护
**为什么不是更大?**
- 0.0: 无SOP，旧类可能下降更多
- 3.0: ✅ 轻度保护，主要依赖自适应性
- 5.0: 中度保护
- 10.0: 强保护，但会抑制新类学习

**SOP的优势**: 不是统一保护所有旧类，而是根据稳定性调整

#### sop_lambda_max: 0.8
```python
sop_lambda_max=0.8  # 高稳定类的最大正则化强度
```
**作用**: 限制最强保护的强度
**为什么是0.8而不是2.0?**
- 我们的目标是鼓励新类发现
- 即使是最稳定的旧类，也不要过度保护
- 0.8是温和的保护强度

#### sop_lambda_min: 0.05
```python
sop_lambda_min=0.05  # 低稳定类的最小正则化强度
```
**作用**: 允许不稳定的旧类充分适应新数据
**为什么是0.05?**
- 0.05 ≈ 几乎不保护
- 对于与新类混淆的旧类，应该允许它们调整边界

#### 其他SOP参数
```python
sop_momentum=0.9          # 稳定性得分平滑更新
sop_use_entropy=True      # 使用熵度量（更稳定）
sop_update_freq=5         # 每5个epoch更新（适中频率）
sop_log_stability=True    # 记录日志便于分析
```

## 📊 预期效果对比

### 原始配置（参考命令）
```
memax_old_new_weight=1, proto_aug_weight=1.0, feat_distill_weight=1.0
lr=0.01, sop_weight=0.0

结果:
Old: 85.0%  (过度保护)
New: 55.0%  (探索不足)
All: 68.0%
```

### 当前配置（鼓励新类发现）
```
memax_old_new_weight=6.0, proto_aug_weight=0.4, feat_distill_weight=0.4
lr=0.015, sop_weight=3.0, sop_lambda_max=0.8, sop_lambda_min=0.05

预期结果:
Old: 79-81%  (降低4-6%，可接受)
New: 62-64%  (提升7-9%，目标达成!) ✅
All: 69-71%  (整体提升1-3%)
```

## 🔄 参数间的协同作用

### 保护机制的总强度
```
总保护 = ProtoAug × 0.4 + FeatDistill × 0.4 + SOP × 3.0 × λ(s_k)

原版: ProtoAug × 1.0 + FeatDistill × 1.0 + 0
     = 2.0 (统一强保护)

当前: ProtoAug × 0.4 + FeatDistill × 0.4 + SOP × 3.0 × (0.05~0.8)
     = 0.8 + 0.15~2.4 (自适应，总体减弱)
```

### 新类学习信号的总强度
```
新类信号 = memax_old_new × 6.0 + memax_new_in × 2.5 + lr × 0.015

原版: 1.0 + 1.0 + 0.01 = 2.01
当前: 6.0 + 2.5 + 0.015 ≈ 8.5 (大幅增强!)
```

**结论**: 保护减弱 + 新类信号增强 = 平衡改善

## 🎚️ 调优空间

如果运行后发现需要调整：

### 场景1: 新类还是不够高
```python
# 进一步减弱保护
proto_aug_weight=0.3
feat_distill_weight=0.3
sop_weight=2.0

# 进一步增强新类信号
memax_old_new_weight=8.0
memax_new_in_weight=3.0
lr=0.02
```

### 场景2: 旧类掉太多
```python
# 增强保护
proto_aug_weight=0.5
feat_distill_weight=0.5
sop_weight=4.0
sop_lambda_max=1.2

# 略微减弱新类信号
memax_old_new_weight=4.0
lr=0.012
```

### 场景3: 平衡良好，想进一步微调
```python
# 只调整SOP
sop_weight=3.5           # 略微增强
sop_lambda_max=1.0       # 提高上限
```

## 🧪 验证指标

运行后检查以下指标：

### 1. 损失值变化
```bash
grep "me_max_loss_old_new" logs/log.txt | head -20
```
**期望**: 从0.01-0.02增加到0.08-0.12

### 2. 预测比例
```bash
grep "Pred new ratio" logs/log.txt | tail -20
```
**期望**: 
```
Pred new ratio: 0.18-0.20 | Ground-truth new ratio: 0.20
(接近ground-truth，说明平衡改善)
```

### 3. SOP Loss
```bash
grep "sop_loss" logs/log.txt | head -20
```
**期望**: 
```
sop_loss: 0.0015-0.0040 (有值，说明SOP在工作)
```

### 4. 稳定性得分
```bash
grep "Stability scores" logs/log.txt | head -5
```
**期望**: 
```
Stability scores: 0.85 0.72 0.91 0.65 ...
(有明显差异，说明SOP能区分不同类的稳定性)
```

### 5. 最终准确率
```bash
tail -100 logs/log.txt | grep "Test Accuracies (Hard)"
```
**期望**: 
```
Old: 79-81% (降低4-6%)
New: 62-64% (提升7-9%)
All: 69-71% (整体提升)
```

## 🚀 快速开始

### 使用默认配置（推荐）
```bash
# 直接运行，使用set_defaults中的配置
python base.py
```

### 覆盖特定参数
```bash
# 如果想测试更激进的配置
python base.py \
    --memax_old_new_weight 8.0 \
    --sop_weight 2.0
```

### 禁用SOP（对比实验）
```bash
# 测试无SOP的效果
python base.py --sop_weight 0.0
```

## 📊 与参考命令的对比

### 参考命令
```bash
CUDA_VISIBLE_DEVICES=0 python train_happy.py \
    --lr 0.01 \
    --memax_old_new_weight 1 \
    --memax_old_in_weight 1 \
    --memax_new_in_weight 1 \
    --proto_aug_weight 1 \
    --feat_distill_weight 1
```

### 当前配置（base.py defaults）
```bash
python base.py
# 等价于:
# --lr 0.015 (+50%)
# --memax_old_new_weight 6.0 (+500%) ⭐
# --memax_new_in_weight 2.5 (+150%)
# --proto_aug_weight 0.4 (-60%)
# --feat_distill_weight 0.4 (-60%)
# --sop_weight 3.0 (新增) ⭐
```

**关键差异**: 
1. **大幅增强新旧平衡** (6倍)
2. **显著减弱保护** (降到40%)
3. **引入自适应保护** (SOP)

## 💡 设计理念

### 为什么这样配置？

1. **不是完全取消保护**
   - proto/feat_distill降到0.4而不是0
   - SOP提供基础的自适应保护
   - 目标: 旧类降低5%以内

2. **强烈鼓励新类探索**
   - memax_old_new从1到6 (强制平衡)
   - memax_new_in从1到2.5 (内部均匀)
   - lr略微提高 (更强适应性)

3. **SOP的智能性**
   - 不是统一保护所有旧类
   - 稳定的旧类: λ≈0.8 (适度保护)
   - 不稳定的旧类: λ≈0.05 (几乎不保护)
   - **让混淆的旧类自己去适应新类边界**

## 🎯 总结

当前配置是**专门为提升新类发现能力设计的**：

✅ 减弱总体保护 (0.4, 0.4, 3.0×λ)
✅ 增强新类信号 (6.0, 2.5)
✅ 智能自适应 (SOP: 0.05~0.8)
✅ 平衡权衡 (旧类略降，新类大升，整体提升)

**预期**: Old 80%, New 63%, All 70% 🎉

