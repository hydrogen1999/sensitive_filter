from transformers import AutoModel
import gdown

AutoModel.from_pretrained("vinai/phobert-base")

# Download the sensitive model
url = 'https://drive.google.com/u/0/uc?id=1P_x0jQlxTrbeOCatv2xYtDqMyPfgrxLk&export=download'
output = './checkpoints/phobert_fold3.pth'
gdown.download(url, output, quiet=False)

# Download tokenizer
url = 'https://drive.google.com/u/0/uc?id=1be8XYOB3Vz7QPuWIDG5hpwNz5GYJXquE&export=download'
output = 'tokenizer.pickle'
gdown.download(url, output, quiet=False)
