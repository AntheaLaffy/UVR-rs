UVR 本地音轨工作台 — Linux x86_64 验证版

解压整个 UVR 文件夹，运行其中的 uvr-gui。桌面运行需要 GTK 3、WebKitGTK 4.1 等 Tauri 系统库；此压缩包不替代系统运行库安装。

模型是独立文件，不在 GUI 二进制中。程序优先使用旁边的 models 文件夹；轻量包的 models 只有说明文件，可在界面“模型管理”中按需下载。包含部分或全部模型的压缩包使用相同目录结构。

也可在“模型目录”中选择其他磁盘上的文件夹或目录软链接。自定义选择会记住；“使用默认目录”可恢复便携目录。开发目录中已有模型时，程序也支持从工作目录识别。

下载可选 Hugging Face 或 GitHub；代理地址支持 HTTP、HTTPS、SOCKS5、SOCKS5H，留空使用系统／环境代理。程序检测完整文件的大小和 SHA-256，下载未完成或校验失败的文件不会标记为可用。取消下载会清理本次临时文件。点击“重新下载”会在新文件校验通过后替换损坏的普通模型文件。

支持的四个文件：
model_bs_roformer_ep_368_sdr_12.9628.ckpt
5_HP-Karaoke-UVR.pth
6_HP-Karaoke-UVR.pth
UVR-DeEcho-DeReverb.pth

音频输入支持 WAV、FLAC、MP3；输出为 44.1 kHz 双声道浮点 WAV，保留增益，同名输出不会覆盖。当前为 CPU 验证版，处理耗时取决于模型和音频长度。运行时无需 Python。

下载来源记录在 model-downloads.json。程序摘要在 SHA256SUMS，附带模型的摘要在 models/SHA256SUMS。
