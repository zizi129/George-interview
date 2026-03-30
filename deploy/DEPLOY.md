# 生产环境部署指南

## 一、上线前必做清单

### 1. 生成强 JWT 密钥

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(48))"
```

将输出结果填入 `.env`：

```
JWT_SECRET=<上面生成的随机字符串>
```

---

### 2. 配置真实短信验证码

#### 方案 A：腾讯云短信（推荐，可复用已有 TTS 密钥）

1. 登录 [腾讯云短信控制台](https://console.cloud.tencent.com/smsv2)
2. 创建短信应用，获得 **SDKAppID**（格式 `14xxxxxxxx`）
3. 申请**短信签名**（需营业执照等，审核 1-2 个工作日）
4. 申请**验证码短信模板**，示例内容：
   ```
   您的验证码为{1}，{2}分钟内有效，请勿泄露给他人。
   ```
   其中 `{1}` = 验证码，`{2}` = 有效分钟数
5. 在 `.env` 中填入：

```env
SMS_DEV_MODE=false
SMS_BACKEND=tencent
TENCENT_SECRET_ID=<your-tencent-secret-id>
TENCENT_SECRET_KEY=<your-tencent-secret-key>
TENCENT_SMS_APPID=<your-sms-app-id>
TENCENT_SMS_SIGN_NAME=你的签名
TENCENT_SMS_TEMPLATE_ID=123456
```

#### 方案 B：阿里云短信

1. 登录 [阿里云短信服务控制台](https://dysms.console.aliyun.com/)
2. 申请**签名**和**模板**，模板示例：
   ```
   您的验证码是${code}，5分钟内有效。
   ```
3. 在 `.env` 中填入：

```env
SMS_DEV_MODE=false
SMS_BACKEND=aliyun
ALIYUN_ACCESS_KEY_ID=xxxxxxxxxxxxxxxx
ALIYUN_ACCESS_KEY_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ALIYUN_SMS_SIGN_NAME=你的签名
ALIYUN_SMS_TEMPLATE_CODE=SMS_xxxxxxxxx
```

4. 安装阿里云 SDK（默认只安装腾讯云）：
   ```bash
   pip install alibabacloud-dysmsapi20170525
   ```

---

### 3. 配置微信支付

1. 在[微信支付商户平台](https://pay.weixin.qq.com/)完成商户入驻
2. 开通 **Native 支付**
3. 在「账户中心 → API 安全」下：
   - 设置 **APIv3 密钥**（32位字符串）
   - 下载 **API 证书**，获得 `apiclient_key.pem` 和证书序列号
4. 在 `.env` 中填入：

```env
WECHAT_MCH_ID=你的商户号
WECHAT_APP_ID=公众号或小程序的AppID
WECHAT_API_KEY=你的APIv3密钥（32位）
WECHAT_NOTIFY_URL=https://your-domain.com/pay/notify
WECHAT_PRIVATE_KEY_PATH=/path/to/apiclient_key.pem
WECHAT_CERT_SERIAL_NO=你的证书序列号
```

> ⚠️ `WECHAT_NOTIFY_URL` 必须是 HTTPS 地址，否则微信无法回调。

---

### 4. 配置 HTTPS（Nginx + Let's Encrypt）

**前提**：服务器有公网 IP，域名已解析到该 IP，80/443 端口已开放。

```bash
# 安装 Nginx 和 Certbot
apt update && apt install -y nginx certbot python3-certbot-nginx

# 复制 Nginx 配置
cp deploy/nginx.conf.example /etc/nginx/sites-available/livetalking
# 编辑配置，将 your-domain.com 替换为实际域名
nano /etc/nginx/sites-available/livetalking

# 启用站点
ln -s /etc/nginx/sites-available/livetalking /etc/nginx/sites-enabled/

# 申请证书（Certbot 会自动修改 nginx 配置）
certbot --nginx -d your-domain.com

# 验证并重载
nginx -t && systemctl reload nginx
```

架构示意：
```
用户浏览器  ──HTTPS──▶  Nginx(:443)  ──HTTP──▶  app.py(:8010)
微信回调    ──HTTPS──▶  Nginx(:443)  ──HTTP──▶  app.py(:8010)/pay/notify
```

---

### 5. 确认 app.py 只监听本地

在 `app.py` 中默认已配置 `127.0.0.1`，外部流量全部由 Nginx 代理：

```python
site = web.TCPSite(runner, '127.0.0.1', opt.listenport)
```

---

## 二、启动服务

```bash
# 复制并编辑 .env
cp .env.example .env
nano .env   # 填入所有生产配置

# 安装依赖（含腾讯云短信 SDK）
pip install -r requirements.txt

# 启动
bash start.sh
```

## 三、验证检查项

| 检查项 | 验证方法 |
|--------|----------|
| JWT_SECRET 已修改 | `grep JWT_SECRET .env` 确认不是 `change-me-...` |
| SMS_DEV_MODE=false | `grep SMS_DEV_MODE .env` |
| 短信可正常发送 | 用真实手机号测试登录 |
| HTTPS 正常 | 浏览器访问 `https://your-domain.com` 显示绿锁 |
| 微信支付回调可达 | 在微信商户平台「开发配置」验证回调地址 |
| app.py 不暴露外网 | `curl http://服务器IP:8010` 应超时或拒绝 |
