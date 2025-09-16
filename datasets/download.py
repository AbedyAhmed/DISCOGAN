import os
import argparse
import zipfile
import tarfile
import shutil
from tqdm import tqdm


# ----------- UTILITAIRE POUR EXTRACTION -----------

def unzip(filepath, target_dir):
    with zipfile.ZipFile(filepath, 'r') as zf:
        zf.extractall(target_dir)
    print(f"✅ Extracted {filepath} into {target_dir}")


def untar(filepath, target_dir):
    with tarfile.open(filepath, 'r:gz') as tf:
        tf.extractall(target_dir)
    print(f"✅ Extracted {filepath} into {target_dir}")


def create_subset(src_dir, dst_dir, n=10000):
    """Copie seulement n fichiers d’un dataset"""
    os.makedirs(dst_dir, exist_ok=True)
    images = sorted(os.listdir(src_dir))[:n]
    for img in tqdm(images, desc=f"Copying {n} files"):
        shutil.copy(os.path.join(src_dir, img), os.path.join(dst_dir, img))
    print(f"✅ Created subset with {len(images)} files at {dst_dir}")


# ----------- CELEBA -----------

def download_celeb_a(dirpath, subset_size=None):
    data_dir = os.path.join(dirpath, 'celebA')
    if os.path.exists(data_dir):
        print('✅ Found CelebA - skip')
        return

    os.makedirs(dirpath, exist_ok=True)

    url = "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip"
    zip_path = os.path.join(dirpath, "celeba.zip")

    print("⬇️ Downloading CelebA (this may take a while)...")
    os.system(f"wget {url} -O {zip_path}")

    unzip(zip_path, dirpath)

    # Subset
    if subset_size is not None:
        subset_dir = os.path.join(dirpath, f"celebA_subset_{subset_size}")
        create_subset(os.path.join(dirpath, "img_align_celeba"), subset_dir, subset_size)


def download_facescrub(dirpath, subset_size=None):
    data_dir = os.path.join(dirpath, 'facescrub')
    os.makedirs(data_dir, exist_ok=True)

    url = "https://raw.githubusercontent.com/daviddao/facescrub-dataset/master/facescrub_actors.txt"
    os.system(f"wget {url} -O {os.path.join(data_dir, 'facescrub_actors.txt')}")
    url = "https://raw.githubusercontent.com/daviddao/facescrub-dataset/master/facescrub_actresses.txt"
    os.system(f"wget {url} -O {os.path.join(data_dir, 'facescrub_actresses.txt')}")

    print("⚠️ Facescrub only provides metadata (lists of URLs). You still need to download images manually.")


# ----------- PIX2PIX DATASETS -----------

def download_pix2pix(category, subset_size=None):
    data_dir = os.path.join('./datasets', category)

    if os.path.exists(data_dir):
        print(f"✅ Found {category} - skip download")
    else:
        CMD = f'bash ./datasets/download_pix2pix.sh "{category}"'
        os.system(CMD)

    # Subset
    if subset_size is not None and os.path.exists(data_dir):
        subset_dir = os.path.join('./datasets', f"{category}_subset_{subset_size}")
        if not os.path.exists(subset_dir):
            create_subset(data_dir, subset_dir, subset_size)


# ----------- MAIN -----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", help="Datasets to download: celebA | edges2handbags | edges2shoes")
    parser.add_argument("--subset", type=int, default=None, help="Nombre d’images à conserver (optionnel)")
    args = parser.parse_args()

    if 'celebA' in args.datasets:
        download_celeb_a('./datasets/', subset_size=args.subset)

    if 'edges2handbags' in args.datasets:
        download_pix2pix('edges2handbags', subset_size=args.subset)

    if 'edges2shoes' in args.datasets:
        download_pix2pix('edges2shoes', subset_size=args.subset)
    if 'facescrub' in args.datasets:
        download_facescrub('./datasets/', subset_size=args.subset)
