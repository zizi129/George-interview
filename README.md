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

## 环境要求

- 推荐：**Ubuntu 24.04**，**Python 3.10**，**CUDA** 与 **PyTorch** 版本匹配（可参考 PyTorch 官网按本机 CUDA 选型）  
- GPU：wav2lip 建议 **NVIDIA 3060** 及以上；musetalk 对显卡要求更高  
- 网络：WebRTC 场景下服务端需开放 **TCP 监听端口**，并保证 **UDP** 可达（或按实际部署配置 TURN/SRS）

---

## 快速开始

### 1. 获取代码

```bash
git clone <你的仓库地址>
cd <项目根目录>
```

### 2. Conda 与 Python 依赖

```bash
conda create -n nerfstream python=3.10
conda activate nerfstream
# 按本机 CUDA 安装 PyTorch，示例（CUDA 12.4）：
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
```

若访问 Hugging Face 不稳定，可在运行前执行：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 3. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`：至少配置 **LLM**（如 `LLM_API_KEY` / `OPENAI_API_KEY`）、**JWT_SECRET**，以及计划使用的 **TTS**、支付等项。**不要将填好的 `.env` 提交到 Git。**

### 4. 模型与数字人形象（不入库，需本地下载）

`models/` 与 `data/avatars/` 体积较大，仓库中不包含。请准备：

1. **口型模型**：例如将 `wav2lip256.pth` 放到 `models/` 并重命名为 `wav2lip.pth`（文件名以实际代码与配置为准）。  
2. **形象包**：将官方提供的 avatar 压缩包解压到 `data/avatars/`，并使 `.env` 中的 `AVATAR_ID` 与目录名一致。

（社区常用的网盘汇总可参考各类数字人项目公开说明；以你实际使用的权重与形象包文档为准。）

### 5. 启动服务

```bash
bash start.sh
```

在浏览器访问：

```text
http://<服务器IP>:<端口>/interview.html
```

端口以 `start.sh` 或 `.env` 中 `LISTEN_PORT` 为准。停止服务可使用 `bash stop.sh`。

---

## 仓库中不会提交的内容

以下路径/类型已在 `.gitignore` 中排除，克隆后需自行准备或本地生成：

| 类型 | 说明 |
|------|------|
| `.env` | 密钥与本地配置 |
| `models/` | 深度学习权重 |
| `data/`（含 `users.db` 等） | 形象资源与本地数据库 |
| 常见视频扩展名 | 如 `**/*.mp4` 等 |
| `ssl/*.pem`、`ssl/*.key` | TLS 证书与私钥 |

`ssl/` 目录说明见 `ssl/README.md`。

---

## 开源许可

本项目在 **Apache License 2.0** 下分发，见仓库根目录 `LICENSE`。部分底层实现衍生自社区开源数字人相关代码；你在分发与修改时，请遵守许可证中关于保留声明与归属的要求。

---

## 品牌资源

- 横向 Logo：`assets/brand-logo-horizontal.png`  
- 矢量图标：`assets/brand-logo.svg`  

页面中使用的品牌图与 `web/` 下资源保持一致时，可与上述文件同步维护。
