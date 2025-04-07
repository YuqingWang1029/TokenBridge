import os
from tqdm import tqdm
import requests


def download_pretrained_vae(overwrite=False):
    download_path = "pretrained_models/vae/kl16.ckpt"
    if not os.path.exists(download_path) or overwrite:
        headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
        os.makedirs("pretrained_models/vae", exist_ok=True)
        r = requests.get("https://www.dropbox.com/scl/fi/hhmuvaiacrarfg28qxhwz/kl16.ckpt?rlkey=l44xipsezc8atcffdp4q7mwmh&dl=0", stream=True, headers=headers)
        print("Downloading KL-16 VAE...")
        with open(download_path, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=1024*1024), unit="MB", total=254):
                if chunk:
                    f.write(chunk)


if __name__ == "__main__":
    download_pretrained_vae()
