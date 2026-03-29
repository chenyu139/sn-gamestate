# sn-gamestate 工作说明、使用说明、训练说明与标注方案

## 1. 文档目的

这份文档整理了本次在 `sn-gamestate` 仓库中已经完成的工作，说明当前环境如何使用、当前结果如何复现、如果后续需要继续训练该体系应该怎么推进，以及如果要采集和标注自己的数据应该怎么组织与落地。

本文档面向两类使用场景：

1. **直接复用当前环境和结果**，快速得到足球比赛的纯 minimap 视频
2. **继续研发**，包括替换模块、训练子模型、采集自己的标注数据

---

## 2. 本次目标与最终结果

本次目标是把 SoccerNet Game State Reconstruction 基线在当前机器上跑通，并把输出改造成你要求的形式：

- 只输出顶视角球场背景
- 球员显示为随时间移动的彩色圆点
- 左右队颜色区分明显
- 不显示原始比赛画面
- 不显示检测框、人物轮廓等覆盖层

### 最终结果

已经成功生成并校验通过的预测版 minimap 视频：

- 输出文件：`outputs/sn-gamestate/2026-03-28/19-39-13/visualization/videos/SNGS-021.mp4`
- 编码：H.264
- 分辨率：1920x1080
- 帧率：25 FPS
- 帧数：750
- 时长：30 秒

这份视频是基于：

- 官方数据集：`data/SoccerNetGS`
- 官方 validation tracker state：`pretrained_models/gamestate-prtreid-strongsort-valid-compressed.pklz`
- 已修改过的 minimap 可视化逻辑

---

## 3. 本次我做了什么

下面按“环境 -> 配置 -> 可视化 -> 权重 -> 运行 -> 验证”的顺序整理。

### 3.1 环境与依赖处理

已在项目根目录的虚拟环境中完成基础依赖安装与修复：

- 使用项目根目录 `.venv`
- 安装项目依赖：`pip install -e .`
- 安装 `mmcv==2.0.1`
- 为避免某些环境冲突，将 `transformers` 调整到 `<4.40`

另外还做了几项环境层修复：

- 修复了 TrackLab 插件搜索路径中的 `EntryPoint.dist` 兼容问题
- 处理了 Matplotlib 缓存目录问题，运行时使用 `MPLCONFIGDIR=/tmp/mpl`
- 处理了大文件下载中断与校验失败问题，采用断点续传重新拉取

### 3.2 按你的要求改成“纯 minimap”

为了去掉原始视频画面和框，只保留顶视球场与圆点，我修改了以下内容：

#### 1）新增纯 minimap 可视化类

文件：`sn_gamestate/visualization/pitch.py`

新增 `Minimap` 视觉组件，核心逻辑是：

- 每帧先生成顶视角球场背景
- 只读取 `detections_pred` 中的 `bbox_pitch`
- 忽略球
- 根据 `team` 和 `role` 决定圆点颜色
- 在顶视球场上绘制圆点

当前颜色规则：

- `team == left`：红色
- `team == right`：蓝色
- `role == referee`：黄色
- 其他未知：白色

#### 2）导出 Minimap visualizer

文件：`sn_gamestate/visualization/__init__.py`

将 `Minimap` 暴露到可视化模块中，便于 Hydra 配置直接引用。

#### 3）将默认可视化改成“只保留 minimap”

文件：`sn_gamestate/configs/visualization/gamestate.yaml`

原先默认包含：

- frame counter
- pitch overlay
- player ellipse
- radar

现在只保留：

- `minimap`

这一步是纯 minimap 输出的关键。

### 3.3 调整运行配置

文件：`sn_gamestate/configs/soccernet.yaml`

做了以下修改：

- `data_dir` 改成绝对路径：`/home/chenyu/workplace/sn-gamestate/data`
- 保持输出视频打开：`visualization.cfg.save_videos: True`
- 将若干 batch size 调整为更适合当前 24GB GPU 的值
- 当前默认只处理 1 个视频：`dataset.nvid: 1`
- 当前默认处理 `valid` 集

这意味着当前仓库状态更适合“先快速跑通一个 clip 并生成 minimap”。

### 3.4 按你的 SSH 规则调整依赖来源

文件：`pyproject.toml`

把这两个依赖改成了 SSH 形式：

- `prtreid @ git+ssh://git@github.com/VlSomers/prtreid.git`
- `torchreid @ git+ssh://git@github.com/VlSomers/bpbreid.git`

这样符合你要求的 GitHub SSH-only 使用方式。

### 3.5 下载并校验了关键模型与状态文件

本次已经完成并验证以下关键文件：

#### 1）PRTReid 权重

- 文件：`pretrained_models/reid/prtreid-soccernet-baseline.pth.tar`
- 期间遇到过 MD5 校验失败，已删除损坏文件并重新下载

#### 2）官方 validation tracker state

- 文件：`pretrained_models/gamestate-prtreid-strongsort-valid-compressed.pklz`
- 来源使用 README 中更新后的 Zenodo 地址
- 文件已完整下载并检查，内部包含 `summary.json` 与视频级 pickle

这份 tracker state 很重要，因为它允许在不重新跑完整检测/跟踪链路的情况下，直接对官方预测结果进行可视化。

### 3.6 排查并绕过了视频写出问题

在当前环境下，直接走 TrackLab 默认可视化流程时，生成的 MP4 出现过：

- `moov atom not found`
- MP4 文件不可播放

我做了两层处理：

#### 第一层：补充释放 writer

在本地虚拟环境内的 TrackLab 安装中，给 `visualization_engine.py` 补上了 `video_writer.release()`，避免文件句柄未正常关闭。

#### 第二层：采用“逐帧渲染 + ffmpeg 封装”的可靠路径

最终已验证可用的产出方式是：

1. 从官方 tracker state 读取预测结果
2. 用当前 minimap visualizer 逐帧渲染 JPG
3. 用 `ffmpeg` 将这些帧重新编码成 MP4

也就是说：

- **当前 minimap 逻辑本身是正确的**
- **最终视频已经生成成功**
- **最稳妥的复现方式是逐帧导出再用 ffmpeg 封装**

### 3.7 对结果做了实际验证

已做过以下验证：

- 代码可编译：对关键可视化模块执行过 `py_compile`
- 输出视频存在且文件大小正常
- 用 `ffprobe` 验证过编码、分辨率、时长、帧数
- 已确认输出满足“纯 minimap”需求

---

## 4. 当前仓库中与本次工作最相关的文件

### 4.1 代码与配置

- `pyproject.toml`
- `sn_gamestate/configs/soccernet.yaml`
- `sn_gamestate/configs/visualization/gamestate.yaml`
- `sn_gamestate/visualization/__init__.py`
- `sn_gamestate/visualization/pitch.py`

### 4.2 环境内补丁

- `.venv/lib/python3.9/site-packages/tracklab/visualization/visualization_engine.py`

说明：

- 这是**本地虚拟环境里的 TrackLab 安装文件**
- 如果你重建虚拟环境，这个补丁需要重新确认是否还在

### 4.3 产物与数据

- 数据集：`data/SoccerNetGS`
- 官方 tracker state：`pretrained_models/gamestate-prtreid-strongsort-valid-compressed.pklz`
- minimap 视频：`outputs/sn-gamestate/2026-03-28/19-39-13/visualization/videos/SNGS-021.mp4`

---

## 5. 你现在应该怎么使用

这一节分成两种情况：

1. **只想直接用当前结果**
2. **想在当前环境上继续复现或扩展**

### 5.1 直接查看当前已经生成的视频

直接打开：

```bash
/home/chenyu/workplace/sn-gamestate/outputs/sn-gamestate/2026-03-28/19-39-13/visualization/videos/SNGS-021.mp4
```

### 5.2 激活当前环境

```bash
cd /home/chenyu/workplace/sn-gamestate
source .venv/bin/activate
export MPLCONFIGDIR=/tmp/mpl
```

### 5.3 用当前配置跑一个 baseline

如果你想先沿用当前配置跑默认单视频：

```bash
cd /home/chenyu/workplace/sn-gamestate
source .venv/bin/activate
export MPLCONFIGDIR=/tmp/mpl
uv run tracklab -cn soccernet
```

说明：

- 当前 `soccernet.yaml` 里默认 `dataset.nvid: 1`
- 默认处理 `valid` 集的第一个视频
- 当前仓库的“纯 minimap 配置”已经生效

### 5.4 更稳妥的预测版 minimap 复现方式

如果你要稳定复现我已经验证成功的那条路径，建议继续使用：

- 官方 tracker state
- 逐帧渲染
- ffmpeg 合成视频

#### 第一步：用 tracker state 渲染帧

在运行目录执行：

```bash
cd /home/chenyu/workplace/sn-gamestate
source .venv/bin/activate
export MPLCONFIGDIR=/tmp/mpl

python - <<'PY'
from pathlib import Path
import os
import cv2
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tracklab.datastruct import TrackerState
from tracklab.pipeline import Pipeline
from tracklab.visualization.visualization_engine import get_group

run_dir = Path('/home/chenyu/workplace/sn-gamestate/outputs/sn-gamestate/2026-03-28/19-39-13')
os.chdir(run_dir)

cfg = OmegaConf.load('configs/config.yaml')
tracking_dataset = instantiate(cfg.dataset)
tracking_set = tracking_dataset.sets[cfg.dataset.eval_set]
tracker_state = TrackerState(
    tracking_set,
    load_file='/home/chenyu/workplace/sn-gamestate/pretrained_models/gamestate-prtreid-strongsort-valid-compressed.pklz',
    save_file=None,
    pipeline=Pipeline(models=[]),
)
vis = instantiate(cfg.visualization)

video_id = list(tracker_state.video_metadatas.index)[0]
with tracker_state(video_id) as ts:
    detections, image_preds = ts.load()
    image_metadatas = ts.image_metadatas[ts.image_metadatas.video_id == video_id]
    image_gts = ts.image_gt[ts.image_gt.video_id == video_id]
    pred_groups = detections.groupby('image_id')
    gt_groups = ts.detections_gt.groupby('image_id')

    for visualizer in vis.visualizers.values():
        try:
            visualizer.preproces(detections, ts.detections_gt, image_preds, ts.image_gt)
        except Exception:
            pass

    frames_dir = Path('/tmp/sn_gamestate_frames_SNGS_021')
    frames_dir.mkdir(parents=True, exist_ok=True)

    for i, image_id in enumerate(image_metadatas.index, start=1):
        frame = vis.draw_frame(
            image_metadatas.loc[image_id],
            get_group(pred_groups, image_id),
            get_group(gt_groups, image_id),
            image_preds.loc[image_id],
            image_gts.loc[image_id],
            len(image_metadatas),
        )
        cv2.imwrite(str(frames_dir / f'{i:06d}.jpg'), frame)

    print(frames_dir)
PY
```

#### 第二步：把帧封装成 MP4

```bash
ffmpeg -y \
  -framerate 25 \
  -i /tmp/sn_gamestate_frames_SNGS_021/%06d.jpg \
  -c:v libx264 \
  -pix_fmt yuv420p \
  /home/chenyu/workplace/sn-gamestate/outputs/sn-gamestate/2026-03-28/19-39-13/visualization/videos/SNGS-021.mp4
```

#### 第三步：校验视频

```bash
ffprobe -v error \
  -select_streams v:0 \
  -show_entries stream=codec_name,width,height,r_frame_rate,nb_frames,duration \
  -of default=noprint_wrappers=1 \
  /home/chenyu/workplace/sn-gamestate/outputs/sn-gamestate/2026-03-28/19-39-13/visualization/videos/SNGS-021.mp4
```

### 5.5 如果想处理更多视频

编辑 `sn_gamestate/configs/soccernet.yaml`：

```yaml
dataset:
  nvid: -1
```

或者只指定某几个视频：

```yaml
dataset:
  vids_dict:
    valid: ['SNGS-021', 'SNGS-022']
```

### 5.6 如果想换成 train / test / challenge

改这里：

```yaml
dataset:
  eval_set: "train"
```

可选值：

- `train`
- `valid`
- `test`
- `challenge`

注意：

- `test` / `challenge` 的官方标签和评估逻辑与 `valid` 不同
- 如果只是为了看 minimap，建议先在 `valid` 集验证流程

---

## 6. 当前结果的局限与已知问题

### 6.1 已经解决的问题

- 依赖安装与版本冲突
- 大模型与 tracker state 的断点续传
- minimap 样式改造
- 预测版 minimap 视频成功生成

### 6.2 还没有彻底收尾的问题

#### 1）CLI 全预测管线仍存在 GPU / 设备识别不稳定问题

现象：

- 有时 venv 内对 NVML / GPU 的探测不稳定
- 系统 Python 与 venv Python 的 CUDA 可见性表现并不完全一致

影响：

- 不影响当前已经完成的 tracker state 可视化路径
- 但如果你后续要完整重跑全检测、全跟踪、全识别链路，建议继续专项排查

#### 2）默认 MP4 写出链路在当前环境下不够稳定

现象：

- 直接输出视频时曾出现损坏 MP4

建议：

- 继续优先使用“逐帧渲染 + ffmpeg 封装”

---

## 7. 怎么训练

这一部分需要先说清楚一个事实：

### 7.1 这个仓库并不是“一个命令端到端训练全部模块”的项目

`sn-gamestate` 本质上是一个**TrackLab 上层任务仓库**，把多个子模块串成完整 Game State Reconstruction 管线。

这里面包含的子问题很多：

- 检测
- ReID / 跟踪
- 球场线定位 / 相机标定
- 球衣号码识别
- 队伍归属
- 队伍左右侧判断

这些模块里，真正的训练入口很多并不统一放在这个仓库中，而是来自：

- TrackLab
- Ultralytics / YOLO
- PRTReid / BPBreID
- MMOCR / EasyOCR
- TVCalib / PnLCalib / NBJW 等上游项目

所以更现实的训练方式是：

> **按模块训练，再把训练好的模型接回 TrackLab 管线。**

### 7.2 推荐的训练顺序

如果你要继续研发，我建议按下面顺序推进：

#### 第一阶段：先保证推理和评估闭环

先确保你能稳定完成：

1. 读取数据
2. 跑单模块
3. 生成 tracker state
4. 输出 minimap
5. 做 valid 集评估

这一阶段当前已经接近完成，唯一还需要继续啃的是“完整 CLI 全预测链路的设备问题”。

#### 第二阶段：优先训练最影响结果的模块

通常优先级建议如下：

1. **检测模块**
2. **ReID / track 模块**
3. **jersey number 模块**
4. **calibration / pitch 模块**
5. **team / team_side / tracklet aggregation**

### 7.3 各模块怎么训练

#### A. 检测模块

当前配置中检测器来自：

- `modules/bbox_detector: yolo_ultralytics`

这类模块一般不在本仓库内做训练脚本封装，通常使用 YOLO 自己的训练流程。

训练思路：

1. 先从 `Labels-GameState.json` 中导出 `bbox_image`
2. 转成 YOLO 所需格式
3. 训练检测器
4. 导出 best checkpoint
5. 在 TrackLab 配置里把检测模型路径替换掉

你需要保留的关键监督字段：

- `bbox_image`
- 类别（至少 person；如果你想显式识别球，也可以单独建 ball 类）

#### B. ReID / team / role 模块

当前配置中 ReID 主要依赖：

- PRTReid
- BPBreID 相关组件

训练思路：

1. 从每个 `bbox_image` 裁剪人物 patch
2. 使用 `track_id` 作为身份监督
3. 使用 `attributes.role` 作为角色监督
4. 使用 `attributes.team` 作为队伍监督
5. 训练完成后替换 ReID 权重

这一步通常对最终 minimap 中“同一人是否保持稳定身份”“左右队是否稳定”影响很大。

#### C. 球衣号码识别模块

当前基线中用的是：

- `modules/jersey_number_detect: mmocr`

训练思路：

1. 从球员框中截取上身或号码区域
2. 以 `attributes.jersey` 作为文本标签
3. 训练检测 + 识别，或者直接训练号码识别器
4. 回接到 MMOCR 推理接口

注意：

- 裁剪质量非常关键
- 模糊、遮挡、运动拖影会显著影响号码识别质量

#### D. 球场定位 / 标定模块

当前配置已经切到：

- `modules/pitch: nbjw_calib`
- `modules/calibration: nbjw_calib`

这类模块的训练通常依赖：

- 球场线标注
- 相机参数
- 或者关键点 / 线段监督

如果你要自己训练，通常需要参考这些上游项目各自的训练方式，而不是仅依赖 `sn-gamestate` 仓库本身。

#### E. team_side / tracklet aggregation

这类模块多为规则型、聚合型、轻学习型模块。

例如：

- `tracklet_agg` 会做轨迹级投票
- `team_side` 会根据位置推断左右侧

这类模块通常不是最先训练的对象，更适合在前面几个核心模块稳定以后再微调策略。

### 7.4 训练时最务实的项目策略

如果你要把这套系统真正迭代起来，建议按下面方式执行：

#### 路线 1：只替换一个模块

例如只换检测器：

1. 用自己的检测模型训练
2. 只替换 bbox detector
3. 其余模块沿用现有基线
4. 在 valid 集做消融和对比

优点：

- 开发成本低
- 问题定位清楚

#### 路线 2：只替换识别类模块

例如只换：

- ReID
- jersey number
- team affiliation

优点：

- 不动前面检测模块
- 更容易观察身份与号码精度收益

#### 路线 3：构建自己的全流程

这是成本最高的路线，建议在你已经稳定掌握数据格式、模块接口与评估方法之后再做。

---

## 8. 如果要收集和标注自己的数据，应该怎么做

这一节分成两个前提：

### 8.1 如果你是为了参加官方 benchmark / challenge

要注意 `ChallengeRules.md` 中的限制：

- 比赛期内不允许通过额外人工标注制造不公平优势
- 用于训练和评估的数据需要公开可获取
- 不应在挑战期间手工修改官方测试标签

所以：

- **如果你的目标是遵守官方 benchmark 规则，就不要随意加私有人工标注数据去做正式对比提交**

### 8.2 如果你是为了内部项目、科研扩展、产品验证

那就完全可以自己采集和标注，但建议**尽量对齐 SoccerNetGS 的数据组织方式**，这样后续最容易接入当前仓库。

### 8.3 推荐的数据目录结构

建议按官方风格组织：

```text
your_dataset/
  train/
    CLIP-001/
      img1/
        000001.jpg
        000002.jpg
        ...
      Labels-GameState.json
  valid/
    CLIP-101/
      img1/
      Labels-GameState.json
  test/
    CLIP-201/
      img1/
      Labels-GameState.json
```

每个 clip 一个目录，目录中至少包含：

- `img1/`：逐帧图片
- `Labels-GameState.json`：标注文件

### 8.4 建议的采集规范

为了让训练与推理更稳定，建议采集时控制以下因素：

#### 视频层面

- 尽量保持固定帧率，推荐 25 FPS
- 分辨率尽量接近 1920x1080
- 单个 clip 保持 10 到 30 秒，更适合标注与调试
- 尽量覆盖不同机位、变焦、光照、天气、球衣颜色组合

#### 内容层面

要尽量覆盖：

- 开阔视角
- 密集对抗
- 遮挡严重场景
- 快速横移镜头
- 远景与近景
- 定位球、角球、任意球等特殊场景

### 8.5 推荐的标注字段

如果你要兼容当前任务，建议至少标下面这些字段。

#### 每帧级信息

- `image_id`
- `file_name`
- `width`
- `height`
- 是否有人体标注
- 是否有球场标注
- 是否有相机标注

#### 每个实例级信息

- `track_id`
- `bbox_image`
- `bbox_pitch`
- `attributes.role`
- `attributes.team`
- `attributes.jersey`

### 8.6 一个真实样例长什么样

下面是从当前官方数据中抽象出来的结构示意：

```json
{
  "info": {
    "version": "1.3",
    "name": "SNGS-021",
    "im_dir": "img1",
    "frame_rate": 25,
    "seq_length": 750,
    "im_ext": ".jpg"
  },
  "images": [
    {
      "is_labeled": true,
      "image_id": "2021000001",
      "file_name": "000001.jpg",
      "height": 1080,
      "width": 1920,
      "has_labeled_person": true,
      "has_labeled_pitch": true,
      "has_labeled_camera": true
    }
  ],
  "annotations": [
    {
      "id": "2021000001",
      "image_id": "2021000001",
      "track_id": 1,
      "category_id": 1,
      "attributes": {
        "role": "player",
        "jersey": "3",
        "team": "right"
      },
      "bbox_image": {
        "x": 1699,
        "y": 236,
        "w": 33,
        "h": 80
      },
      "bbox_pitch": {
        "x_bottom_middle": 15.09,
        "y_bottom_middle": -20.79
      }
    }
  ]
}
```

### 8.7 标注流程建议

如果你要自己做一套数据，建议按以下顺序。

#### 步骤 1：先切 clip，再抽帧

推荐：

- 先把比赛视频切成短片段
- 每段统一抽帧到 `img1/`
- 保证文件名连续，例如 `000001.jpg`

#### 步骤 2：先标 2D 人框与 track_id

最先做的是：

- 每帧人体框 `bbox_image`
- 跨帧一致的 `track_id`

这是整个系统里最基础的一层，没有它后面很多监督都不稳。

#### 步骤 3：补角色、队伍、号码

对每个 track 或每帧实例补：

- `role`：`player / goalkeeper / referee / other / ball`
- `team`：`left / right / null`
- `jersey`：可见时填数字，不可见时留空或 null

#### 步骤 4：补 pitch 坐标与标定信息

如果你的目标是做真正的 minimap，而不是只做图像域检测，那么还需要：

- 球场线
- 关键点
- 或者相机参数
- 以及每个实例的 `bbox_pitch`

这部分是最难标的，因为它涉及图像到球场平面的映射。

### 8.8 最推荐的落地方法：半自动标注

不要一开始就全手工做。更现实的方法是：

#### 第一轮：自动预标

用现有基线先自动产出：

- bbox
- track_id
- role / team / jersey 初始结果
- pitch 投影初值

#### 第二轮：人工校正

人工只修改错误项：

- 漏检
- 错框
- 身份切换
- 队伍错分
- 号码错误
- pitch 点位明显偏移

#### 第三轮：导回统一 JSON

把修正后的标注重新导出成与你训练脚本兼容的统一格式。

这样效率最高，也最接近实际可维护的数据生产流程。

### 8.9 标注质量检查清单

建议每轮数据都做以下检查：

- `track_id` 是否跨帧稳定
- `bbox_image` 是否紧贴人物
- `jersey` 是否只在可辨认时填写
- `team` 是否始终按照镜头视角的 left/right 定义
- `referee / goalkeeper / player` 是否区分清楚
- `bbox_pitch` 是否落在合理球场范围内
- clip 的帧率、分辨率、文件命名是否统一

---

## 9. 如果你要把自己的数据接进当前仓库

有两条路线。

### 9.1 路线 A：尽量仿照 SoccerNetGS 格式

这是最推荐的路线。

做法：

1. 目录结构模仿 `SoccerNetGS`
2. 标注 JSON 模仿 `Labels-GameState.json`
3. 让字段命名尽量一致
4. 修改 `dataset_path`
5. 必要时在 TrackLab 数据集读取处加少量适配

优点：

- 改动最小
- 最容易沿用现有代码和配置

### 9.2 路线 B：写你自己的数据集适配器

如果你的数据格式和 SoccerNet 差异很大，可以自己写一个新的数据集类和配置项。

这时建议：

- 新建 dataset config
- 明确 image metadata / detection annotation / image gt 的映射关系
- 保证输出字段与现有模块期望字段一致

这种方法更灵活，但开发成本更高。

---

## 10. 后续最值得继续做的事

如果你下一步还要继续推进，我建议优先按下面顺序：

### 第一优先级

- 彻底跑通 **完整 CLI 全预测链路**
- 把当前 GPU / NVML 识别不稳定问题消掉

### 第二优先级

- 把“tracker state -> 帧渲染 -> ffmpeg”的流程做成固定脚本
- 这样以后不需要手工粘贴 heredoc 命令

### 第三优先级

- 选择一个核心子模块开始训练
- 通常优先从检测或 ReID 入手

### 第四优先级

- 为自己的数据建立一份对齐 SoccerNetGS 的数据规范
- 先做 5 到 10 个 clip 的小样本验证集

---

## 11. 一份最实用的结论

如果你现在只关心“这套东西已经到了什么程度”，可以直接看这段：

### 已经完成的

- 环境已基本可用
- 官方数据集已对接
- 关键权重与官方 valid tracker state 已下载完成
- minimap 已按你的要求改成纯顶视图圆点风格
- 已成功生成并验证预测版 MP4

### 现在就能做的

- 直接查看已有 minimap 视频
- 按本文档复现当前 minimap 输出
- 开始基于 SoccerNetGS 结构准备自己的数据

### 还值得继续做的

- 彻底修掉完整 CLI 管线的 GPU 问题
- 把当前验证流程脚本化
- 选择一个模块开始训练迭代

---

## 12. 参考资料

建议后续继续阅读以下材料：

- 仓库 README：`README.md`
- 挑战规则：`ChallengeRules.md`
- 主配置：`sn_gamestate/configs/soccernet.yaml`
- 可视化配置：`sn_gamestate/configs/visualization/gamestate.yaml`
- minimap 实现：`sn_gamestate/visualization/pitch.py`
- 官方论文：<https://arxiv.org/abs/2404.11335>
- TrackLab：<https://github.com/TrackingLaboratory/tracklab>

