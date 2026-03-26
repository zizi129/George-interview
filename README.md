# 乔治面试 · George Interview

[English](./README-EN.md)

<p align="center">
  <img src="./assets/brand-logo-horizontal.png" alt="乔治面试" width="360" />
</p>

<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License" /></a>
  <img src="https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white" alt="Python" />
</p>

**乔治面试**是一套面向招聘与练习场景的 **AI 数字人实时面试** 系统：支持岗位与简历上下文、多轮问答、语音/文本交互（具体能力以当前分支代码为准），通过 WebRTC 推送数字人画面。

---

## 功能概览

- 面试准备页：岗位信息、简历解析与面试模式配置  
- 数字人驱动：基于 wav2lip / musetalk 等口型模型（需自行下载权重与形象包）  
- 账号与额度、支付扩展（见 `.env.example`）  
- Web 前端入口：`/interview.html`

---

## 使用链接

- GitHub 仓库：[`https://github.com/zizi129/George-interview`](https://github.com/zizi129/George-interview)
- 夸克资源包：[`https://pan.quark.cn/s/a86a5631b659`](https://pan.quark.cn/s/a86a5631b659)
- 前端访问地址：[`https://njuai-interview.top:8010/interview.html`](https://njuai-interview.top:8010/interview.html)

> `upload.zip` 为团队内部运行资源包，包含 `.env`、默认 `wav2lip` 模型权重，以及当前前端正在使用的两套数字人形象。请仅在团队内部传阅，不要上传到 GitHub。

---

## 环境要求

- 推荐系统：**Ubuntu 24.04**
- Python：**3.10**
- 建议显卡：**NVIDIA 3060** 及以上
- 默认运行环境名：`nerfstream`
- 默认数字人模型：`wav2lip`
- 当前默认面试官形象：
  - 中文：`wav2lip_avatar_atlas`
  - 英文：`wav2lip_avatar_atlas5_en_v1`

---

## 环境配置教程

### 1. 检查是否已安装 Conda

```bash
conda --version
```

如果提示 `command not found`，请先安装 Miniconda。

### 2. 安装 Miniconda（未安装时执行）

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
source ~/miniconda3/bin/activate
conda init bash
exec bash
```

> 上述步骤完成后，`conda activate nerfstream` 就可以正常使用。

### 3. 创建统一环境 `nerfstream`

```bash
conda create -n nerfstream python=3.10 -y
conda activate nerfstream
```

### 4. 安装 PyTorch 与项目依赖

先确认本机 CUDA 版本：

```bash
nvidia-smi
```

如果本机 CUDA 为 **12.4**，可直接执行：

```bash
conda activate nerfstream
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

如果 CUDA 版本不是 12.4，请按 [PyTorch 官方安装说明](https://pytorch.org/get-started/previous-versions/) 选择匹配版本后，再执行：

```bash
conda activate nerfstream
pip install -r requirements.txt
```

如果访问 Hugging Face 较慢，可在当前终端先执行：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 资源下载与解压

### 1. 克隆项目代码

```bash
git clone https://github.com/zizi129/George-interview.git
cd George-interview
```

### 2. 下载 `upload.zip`

请从夸克网盘下载：

- [`https://pan.quark.cn/s/a86a5631b659`](https://pan.quark.cn/s/a86a5631b659)

> 提取码请联系项目负责人获取。

下载后的文件名保持为 `upload.zip`，并放到项目根目录：

```text
George-interview/upload.zip
```

### 3. 安装解压工具（若未安装）

```bash
sudo apt-get update
sudo apt-get install -y unzip
```

### 4. 在项目根目录解压

```bash
cd George-interview
unzip -o upload.zip
```

### 5. 解压后应存在的关键路径

```text
George-interview/
├── .env
├── models/
│   └── wav2lip.pth
└── data/
    └── avatars/
        ├── wav2lip_avatar_atlas/
        │   ├── coords.pkl
        │   ├── face_imgs/
        │   └── full_imgs/
        └── wav2lip_avatar_atlas5_en_v1/
            ├── coords.pkl
            ├── face_imgs/
            └── full_imgs/
```

### 6. 解压后自检

```bash
cd George-interview
ls -l .env
ls -l models/wav2lip.pth
ls -l data/avatars/wav2lip_avatar_atlas
ls -l data/avatars/wav2lip_avatar_atlas5_en_v1
```

只要上述路径都存在，当前默认运行链路所需资源就已经补齐。

---

## 启动与使用教程

### 每次启动前先执行

```bash
cd George-interview
conda activate nerfstream
sudo service nginx restart
```

> 如果当前机器默认就是 `root`，可以直接执行 `service nginx restart`。

### 首次启动 / 日常重启建议顺序

```bash
cd George-interview
conda activate nerfstream
bash stop.sh || true
sudo service nginx restart
bash start.sh
```

### 浏览器访问地址

[`https://njuai-interview.top:8010/interview.html`](https://njuai-interview.top:8010/interview.html)

---

## 运行前自检

启动前建议确认以下几项：

```bash
conda activate nerfstream
python --version
nvidia-smi
ls models/wav2lip.pth
ls data/avatars/wav2lip_avatar_atlas
ls data/avatars/wav2lip_avatar_atlas5_en_v1
```

如果 `conda activate nerfstream` 无法执行，通常是因为：

1. 还没有安装 Conda  
2. 还没有创建 `nerfstream` 环境  
3. 安装后没有执行 `conda init bash` 并重新打开 shell

---

## 仓库中不会提交的内容

以下内容已在 `.gitignore` 中排除，不会进入 GitHub：

| 类型 | 说明 |
|------|------|
| `.env` | 本地密钥与环境配置 |
| `models/` | 深度学习权重 |
| `data/`（含 `users.db` 等） | 数字人形象资源与本地数据库 |
| 常见视频扩展名 | 如 `**/*.mp4`、`**/*.mov` 等 |
| `ssl/*.pem`、`ssl/*.key` | TLS 证书与私钥 |
| `results/`、日志、临时音频 | 运行时产物 |

当前团队的推荐做法是：

1. 代码放 GitHub  
2. 运行资源放夸克网盘 `upload.zip`  
3. 队友克隆后，在项目根目录解压 `upload.zip` 即可恢复本地运行所需的关键数据

---

## 开发者指南

### 分支规则

- **禁止直接在 `main` 分支开发、提交或发起 PR**
- `main`：稳定版本 / 受控发布使用
- `dev`：日常集成分支，所有功能完成后统一向 `dev` 提 PR
- `feature`：当前仓库的功能基线分支

> 由于远程仓库已经存在名为 `feature` 的分支，为避免 Git 分支命名冲突，功能分支不要使用 `feature/xxx`。  
> 本项目统一使用：`feat-功能名`  
> 例如：`feat-resume-parser`、`feat-payment-ui`、`feat-interview-report`

### 首次同步远程分支

```bash
git clone https://github.com/zizi129/George-interview.git
cd George-interview
git fetch origin
git branch -a
git checkout dev
git pull origin dev
```

### 从 `feature` 基线拉出功能分支

```bash
cd George-interview
git fetch origin
git checkout -b feat-resume-parser origin/feature
```

### 开发与提交

```bash
cd George-interview
git status
git add .
git commit -m "feat: 优化简历解析与画像展示"
git push -u origin feat-resume-parser
```

### 提交 PR 到 `dev`

功能分支推送完成后，到仓库页面发起 Pull Request：

- 仓库地址：[`https://github.com/zizi129/George-interview`](https://github.com/zizi129/George-interview)
- `base` 选择：`dev`
- `compare` 选择：`feat-resume-parser`

**不要向 `main` 发起 PR。**

### 本地分支与远程分支常用命令

查看本地分支：

```bash
git branch
```

查看远程分支：

```bash
git branch -r
```

查看本地 + 远程全部分支：

```bash
git branch -a
```

同步远程信息：

```bash
git fetch origin
```

切换到 `dev` 并拉最新代码：

```bash
git checkout dev
git pull origin dev
```

从远程 `feature` 拉新的功能分支：

```bash
git checkout -b feat-your-topic origin/feature
```

推送本地分支到远程：

```bash
git push -u origin feat-your-topic
```

删除本地已合并分支：

```bash
git branch -d feat-your-topic
```

删除远程已合并分支：

```bash
git push origin --delete feat-your-topic
```

### 推荐开发流程

1. 先 `git fetch origin`
2. 确认 `dev` 与 `feature` 的最新状态
3. 从 `origin/feature` 拉出自己的 `feat-功能名` 分支
4. 在自己的功能分支开发、提交、推送
5. 向 `dev` 发起 PR
6. 合并完成后，删除本地和远程功能分支

---

## 开源许可

本项目在 **Apache License 2.0** 下分发，见仓库根目录 `LICENSE`。部分底层实现衍生自社区开源数字人相关代码；你在分发与修改时，请遵守许可证中关于保留声明与归属的要求。

---

## 品牌资源

- 横向 Logo：`assets/brand-logo-horizontal.png`  
- 标签页图标：`web/brand-logo.png`  

页面中使用的品牌图与 `web/` 下资源保持一致时，可与上述文件同步维护。
