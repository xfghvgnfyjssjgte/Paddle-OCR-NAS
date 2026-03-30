FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# System deps
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    pandoc \
    curl \
    wget \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PaddlePaddle CPU
RUN pip3 install --no-cache-dir --break-system-packages \
    paddlepaddle==3.3.1 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# App dependencies (paddleocr 3.x, 去掉固定版本号)
COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# App code
COPY app.py /app/app.py
COPY s2t_dict.py /app/s2t_dict.py
COPY static/ /app/static/
COPY README.md /readme.md
# 预下载 PP-OCRv5 mobile 模型 + 所有可能用到的子模型
# 全部开启确保方向分类(PP-LCNet)、畸变矫正(UVDoc)、检测(mobile_det)、识别(mobile_rec)均被缓存
# 运行时按前端/环境变量动态决定是否启用各子模型，无需重复下载
RUN python3 -c "\
import os; \
os.environ['CUDA_VISIBLE_DEVICES']='-1'; \
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK']='True'; \
os.environ['FLAGS_use_onednn']='0'; \
os.environ['FLAGS_use_mkldnn']='0'; \
from paddleocr import PaddleOCR; \
ocr = PaddleOCR( \
    text_detection_model_name='PP-OCRv5_mobile_det', \
    text_recognition_model_name='PP-OCRv5_mobile_rec', \
    use_doc_orientation_classify=True, \
    use_doc_unwarping=True, \
    use_textline_orientation=False, \
    device='cpu', \
    enable_mkldnn=False); \
print('PP-OCRv5 mobile models cached.')"

RUN mkdir -p /tmp/pdf_ocr_output /tmp/pdf_ocr_uploads

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

CMD ["python3", "-m", "gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "1", \
     "--threads", "4", \
     "--timeout", "600", \
     "--keep-alive", "5", \
     "app:app"]
