ARG BASE_IMAGE=local/codebuddy2api:f717db6
FROM ${BASE_IMAGE}
WORKDIR /app
# 基础镜像可能早于 requirements.txt 的更新（如 cryptography），这里补齐依赖。
# 国内服务器可用 --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_INDEX_URL=https://pypi.org/simple
COPY requirements.txt ./
RUN pip install --no-cache-dir -i ${PIP_INDEX_URL} -r requirements.txt
COPY core ./core
COPY admin ./admin
CMD ["python3", "-m", "admin.server"]
