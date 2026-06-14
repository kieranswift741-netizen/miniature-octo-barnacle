# 量子电池 LSTM 实验说明

三个实验从不同角度验证 LSTM 模型对量子电池非马尔可夫动力学的预测能力。

---

## 物理背景

### 系统模型

- **N=1 二能级系统 (TLS)**：哈密顿量 $H_S = \frac{\omega_0}{2}\sigma_z + F\sigma_x$
  - $\frac{\omega_0}{2}\sigma_z$：裸系统能量（$\omega_0=1.0$，本征值 $\pm 0.5$）
  - $F\sigma_x$：外部驱动（充电场），$F=1.0$
- **结构水库**：洛伦兹谱密度 $J(\omega) = \frac{1}{2\pi}\frac{\gamma\Omega^2}{(\omega-\omega_c)^2+\gamma^2}$
  - $\Omega$：耦合强度（越大 → 系统-环境相互作用越强）
  - $\gamma$：谱宽度（越小 → 水库记忆时间 $\tau_m=1/\gamma$ 越长 → 非马尔可夫性越强）
- **动力学求解**：QuTiP HEOM（Hierarchical Equations of Motion），截断深度 `max_depth=5`

### 能量定义

| 量 | 公式 | 范围 | 含义 |
|----|------|------|------|
| 裸能量 | $E_{\text{bare}} = \text{Tr}[\rho H_S^{\text{bare}}]$ | $[-0.5, 0.5]$ | 包含负数，基态为负 |
| 储存能量 | $E_{\text{stored}} = E_{\text{bare}} + 0.5$ | $[0, 1.0]$ | 电池语境使用，基态=0，激发态=1 |

> **约定**：$H_S^{\text{bare}} = \frac{\omega_0}{2}\sigma_z = \text{diag}(0.5, -0.5)$。$|0\rangle$ 是激发态（E=+0.5），$|1\rangle$ 是基态（E=-0.5）。电池初始态为 $|0\rangle$（充满，储存能量=1.0）。

### 非马尔可夫性度量

- **记忆时间**：$\tau_m = 1/\gamma$ — 水库关联函数的衰减时间尺度
- **非马尔可夫性**：$\Omega/\gamma$ — 越大表示环境记忆效应越强
- 当 $\gamma$ 很小（$\tau_m$ 很大）时，能量可以从水库回流到系统（energy backflow），这是非马尔可夫动力学的标志

---

## 输出文件说明

每个实验在 `output/<experiment_name>/` 下生成以下文件：

```
output/<experiment>/
├── data/                       # 生成的 HEOM 轨迹数据
│   ├── traj_XXXX.npz           # 单条轨迹（times, features, energies, ergotropies）
│   └── metadata.json           # 轨迹索引和参数
├── models/
│   ├── best_model.pt           # 最佳验证损失模型
│   └── final_model.pt          # 最终模型
├── logs/
│   └── training_history.json   # 训练历史（epoch, train_loss, val_loss, 物理约束分量）
├── experiment_config.yaml      # 实验的完整配置
├── training_history.json       # 训练历史副本
├── evaluation.json             # 评估结果（MSE, MAE, 逐轨迹预测）
└── *.png                       # 可视化图表
```

---

## Experiment A：非马尔可夫增强效应

### 研究问题

> 谱宽度 $\gamma$（即非马尔可夫性强弱）如何影响电池的充电效率？LSTM 能否捕捉到小 $\gamma$ 下的能量回流？

### 实验设置

| 参数 | 值 |
|------|-----|
| 耦合强度 $\Omega$ | [0.1, 0.2] |
| 谱宽度 $\gamma$ | [0.05, 0.1, 0.2, 0.5, 1.0, 2.0] |
| 总演化时间 | 30.0 |
| 训练 epoch | 100 |

共 $2 \times 6 = 12$ 条轨迹（部分 $\gamma=2.0$ 可能因 HEOM 不收敛被自动过滤）。

### 结果解读

评估文件 `evaluation.json` 中每条轨迹包含：

- `energy_mse`：预测能量与真实能量的均方误差（越小越好）
- `energy_mae`：平均绝对误差
- `efficiency_true` / `efficiency_pred`：充电效率 $\eta = E(t)/E_{\text{input}}$（真实/预测）
- `times`、`true_energies`、`predicted_energies`：完整时间序列

关键图表 `efficiency_vs_gamma.png`：
- **左图**：横轴 $\gamma$（从右到左减小，非马尔可夫性增强），纵轴充电效率
- **右图**：横轴 $\gamma$，纵轴预测 MSE
- **期望现象**：小 $\gamma$（强非马尔可夫性）→ 效率可能更高（能量回流辅助充电），MSE 可能更大（动力学更复杂）

### 评判标准

| 指标 | 优秀 | 良好 | 需改进 |
|------|------|------|--------|
| Energy MSE | < 0.005 | 0.005-0.02 | > 0.02 |
| 效率预测趋势 | 与真实一致 | 方向正确 | 完全错误 |

### 物理启示

如果实验显示小 $\gamma$ 时 LSTM 预测的充电效率高于大 $\gamma$（马尔可夫极限），说明模型成功学到了非马尔可夫环境中的能量回流效应（feedback gain）。这是经典马尔可夫主方程无法捕捉的现象。

---

## Experiment B：存储寿命预测

### 研究问题

> 电池充电后切断驱动（$F=0$），在不同谱宽度 $\gamma$ 的环境下，电池能保持 >90% 峰值能量多长时间？

### 实验设置

| 参数 | 值 |
|------|-----|
| 耦合强度 $\Omega$ | [0.1] |
| 谱宽度 $\gamma$ | [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 2.0] |
| 驱动 | **关闭**（自由衰减） |
| 初始态 | $\vert 0\rangle$（激发态，储存能量=1.0） |
| 总演化时间 | 40.0 |

### 结果解读

评估文件 `shelf_lives.json` 中每条轨迹包含：

- `gamma`：谱宽度
- `memory_time`：记忆时间 $\tau_m = 1/\gamma$
- `shelf_life_true`：真实储存寿命（储存能量降至峰值 90% 以下的时间）
- `shelf_life_pred`：LSTM 预测的储存寿命

关键图表 `shelf_life_vs_gamma.png`：
- 横轴 $\gamma$（从右到左减小），纵轴储存寿命
- **期望现象**：小 $\gamma$ → 强非马尔可夫性 → 能量回流 → 储存寿命更长（能量不是单调衰减，而是振荡衰减）

### 评判标准

| 指标 | 优秀 | 良好 | 需改进 |
|------|------|------|--------|
| shelf_life 预测误差 | < 10% | 10-30% | > 30% |
| $\gamma$-寿命趋势 | 与真实一致 | 大致正确 | 完全错误 |

### 物理启示

经典马尔可夫衰变的储存寿命正比于 $1/\Omega^2$（费米黄金规则），与非马尔可夫性 $\gamma$ 无关。如果实验显示小 $\gamma$ 时储存寿命显著偏离马尔可夫预期，说明 LSTM 学到了非马尔可夫记忆效应对能量保持的增强作用。

---

## Experiment C：跨参数外推

### 研究问题

> LSTM 能否预测训练时**未见过**的耦合强度 $\Omega$ 下的电池行为？这测试模型是"记住"了训练数据，还是学到了底层物理规律。

### 实验设置

| 参数 | 值 |
|------|-----|
| 训练 $\Omega$ | [0.05, 0.1, 0.2, 0.3, 0.4] |
| 测试 $\Omega$ | [0.5, 0.6]（未见过的值） |
| 谱宽度 $\gamma$ | [0.1, 0.3, 0.5, 1.0] |
| 训练 epoch | 150 |

### 结果解读

评估文件 `generalization.json` 包含：

- `in_distribution_mse`：训练分布内的预测 MSE
- `out_of_distribution_mse`：外推到未见 $\Omega$ 的预测 MSE
- `generalization_ratio` = out_mse / in_mse（核心指标）
- `in_dist_results` / `out_dist_results`：逐轨迹详情

关键图表 `generalization_heatmap.png`：
- 热力图展示不同 $(\Omega, \gamma)$ 参数组合下的预测误差
- 训练区域（$\Omega \leq 0.4$）误差应较小
- 外推区域（$\Omega > 0.4$）误差增大程度反映泛化能力

### 评判标准

| generalization_ratio | 含义 |
|---------------------|------|
| < 1.5 | **优秀**：模型学到了底层物理，外推能力很强 |
| 1.5 - 3.0 | **良好**：有一定泛化能力 |
| 3.0 - 10.0 | **一般**：部分泛化，但外推明显变差 |
| > 10.0 | **差**：模型主要靠记忆，未学到可迁移的物理规律 |

### 物理启示

耦合强度 $\Omega$ 决定了系统和环境的相互作用能标。如果 LSTM 能外推到更大的 $\Omega$，说明它学到了密度矩阵演化的**微分方程结构**（类似于学到 $\dot{\rho} = \mathcal{L}[\rho]$ 而非具体的解 $\rho(t)$）。这是从"曲线拟合"到"物理学习"的关键跨越。

---

## 通用故障排查

### HEOM 不收敛

如果数据生成时出现以下警告：
```
Warning: Failed for {...}: Unphysical trajectory detected: max|feature|=XXXXX
```

说明 HEOM 在该参数下未收敛。解决方法：
1. 增大 `config['heom']['max_depth']`（如 5 → 7）
2. 避免 $\gamma$ 过大或 $\Omega$ 过强的参数组合
3. 检查 `time_step` 是否足够小

### 预测误差过大

如果 MSE > 0.1：
1. 检查数据是否被 HEOM 发散污染（查看生成日志中的 Warning）
2. 确认 `max_depth` 足够（建议 ≥ 5）
3. 增加训练轨迹数量（当前 8-12 条，可扩展 param_grid）
4. 检查 `window_size` 是否匹配动力学时间尺度

### JSON 序列化错误

已修复。如果仍然出现，检查 `evaluation.json` 或 `generalization.json` 中是否含有 numpy 类型，运行前确保代码已更新。

---

## 运行命令

```bash
# Experiment A: 非马尔可夫增强
python experiments/experiment_a_feedback.py --config config.yaml --output ./experiments/output/experiment_a

# Experiment B: 存储寿命
python experiments/experiment_b_shelflife.py --config config.yaml --output ./experiments/output/experiment_b

# Experiment C: 跨参数外推
python experiments/experiment_c_extrapolation.py --config config.yaml --output ./experiments/output/experiment_c

# 切换原子数（例如 N=2）
python experiments/experiment_a_feedback.py --num-tls 2 --output ./experiments/output/experiment_a_n2
```

每个实验预计运行时间：3-10 分钟（取决于 HEOM 轨迹数量和训练 epoch）。

### Notebook 中切换原子数

在 `demo.ipynb` 顶部修改 `NUM_TLS` 变量即可：

```python
NUM_TLS = 2    # 从 1 改为 2（或其他值）
```

所有维度（`D=2^N`、`FEAT_DIM=2*D²`、`E_GROUND=-N*ω₀/2`）会自动推导，后续所有 cell 无需手动修改。

> **注意**：N>1 时 Hilbert 空间维度指数增长（N=2→d=4, N=3→d=8），HEOM 模拟时间显著增加。建议 N≥2 时使用较小的 param_grid 和 `max_depth`。
