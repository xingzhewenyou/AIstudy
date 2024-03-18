# 240318
# Ubuntu环境搭建
# 1、iamc 先安装docker命令，iMac的终端运行docker就行
# 2、拉取镜像docker pull ubuntu，这就会自动建立一个docker容器，操作系统是Ubuntu
# 3、运行Ubuntu容器 docker run -it ubuntu，效果就是iMac的终端类似一个Linux
# 4、查看Linux的版本cat /etc/os-release
# PRETTY_NAME="Ubuntu 22.04.4 LTS"
# NAME="Ubuntu"
# VERSION_ID="22.04"
# VERSION="22.04.4 LTS (Jammy Jellyfish)"
# VERSION_CODENAME=jammy
# ID=ubuntu
# ID_LIKE=debian
# HOME_URL="https://www.ubuntu.com/"
# SUPPORT_URL="https://help.ubuntu.com/"
# BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
# PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
# UBUNTU_CODENAME=jammy
#
# 5、运行iMac搭建好的Ubuntu容器  docker exec -it llama2 /bin/bash
# 6、在Ubuntu容器中安装docker（之前的docker是iMac的）
#
# 6-1、进入Ubuntu容器，升级工具                 apt update
# 6-2、确保从 Docker 而不是从 Ubuntu 仓库安装 Docker      apt-cache policy docker-ce
# 6-3、安装 Docker       apt install docker-ce
# 报错Package 'docker-ce' has no installation candidate    docker无法安装docker，除非docker-in-docker，考虑虚拟机
#
#
#
#
# 尝试在iMac直接安装Ubuntu，不经过iMac的docker
# 1、下载镜像https://ubuntu.com/download/desktop
# 2、VMware安装 ISO文件
# 3、打开terminal界面，能mkdir，表示安装成功
# 4、安装docker
#
# 要在 Ubuntu 上安装 Docker，可以按照以下步骤操作：
#
# 更新 apt 包列表：
#
# 在终端中运行以下命令，以确保你的系统上的 apt 包列表是最新的：
#
# bash
# Copy code
# sudo apt update
# 安装所需的软件包以允许 apt 通过 HTTPS 使用仓库：
#
# bash
# Copy code
# sudo apt install apt-transport-https ca-certificates curl software-properties-common
# 添加 Docker 的官方 GPG 密钥：
#
# bash
# Copy code
# curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
# 设置 Docker 的稳定仓库：
#
# bash
# Copy code
# sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
# 再次更新 apt 包列表：
#
# bash
# Copy code
# sudo apt update
# 确保从 Docker 而不是从 Ubuntu 仓库安装 Docker：
#
# bash
# Copy code
# apt-cache policy docker-ce
# 安装 Docker：
#
# bash
# Copy code
# sudo apt install docker-ce
# 验证 Docker 是否正确安装：
#
# bash
# Copy code
# sudo systemctl status docker
# 这将显示 Docker 服务的状态。如果显示 Active (running)，则表示 Docker 已经成功安装并正在运行。
#
# 现在，Docker 应该已经成功安装在你的 Ubuntu 系统上了。你可以使用 docker 命令来管理容器和镜像。
#
#
# 安装llama2
# https://zhuanlan.zhihu.com/p/645426799
# https://soulteary.com/2023/07/21/use-docker-to-quickly-get-started-with-the-official-version-of-llama2-open-source-large-model.html
# 报错找不到模型文件
# [Errno 2] No such file or directory: '/app/LinkSoul/Chinese-Llama-2-7b'

# File
# "/app/llama.cpp/convert.py", line
# 1311, in < module >
# main()

#
# 下载模型文件
# 申请许可https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
#
#
# 