
# Repository requirements

```python

conda create -n UNI python=3.7 -y
conda activate UNI
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
