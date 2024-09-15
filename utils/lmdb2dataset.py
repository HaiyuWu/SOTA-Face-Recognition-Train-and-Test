from os import path, makedirs
import lmdb
import msgpack
import numpy as np
from PIL import Image
import six
from tqdm import tqdm
import argparse


def extract_lmdb(db_path, output_dir):
    # Open the LMDB environment
    env = lmdb.open(
        db_path,
        subdir=path.isdir(db_path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )

    # Read metadata
    with env.begin(write=False) as txn:
        length = msgpack.loads(txn.get(b"__len__"))
        keys = msgpack.loads(txn.get(b"__keys__"))
        classnum = msgpack.loads(txn.get(b"__classnum__"))

    print(f"Extracted {length} images and labels to {output_dir}")
    print(f"Number of classes: {classnum}")

    dataset_name = path.basename(db_path)[:-5]
    images_dir = path.join(output_dir, f'{dataset_name}')

    # Extract data
    im_num = 0
    cur_id = ""
    for idx in tqdm(range(length), desc="Extracting data"):
        with env.begin(write=False) as txn:
            byteflow = txn.get(keys[idx])

        unpacked = msgpack.loads(byteflow)

        # Extract image
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

        # Extract label
        label = unpacked[1]

        # Save image
        if cur_id != label:
            im_num = 0
            cur_id = label
        sub_folder = f"{images_dir}/{int(label):06d}"
        makedirs(sub_folder, exist_ok=True)
        img_path = path.join(sub_folder, f"{im_num:03d}.jpg")
        img.save(img_path)
        im_num += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .lmdb datasets to image")
    parser.add_argument('--db_path', '-db', help="Path of the .lmdb dataset", type=str)
    parser.add_argument('--dest', '-d', help="Destination", type=str, default='./HSFaces')
    args = parser.parse_args()
    db_path = args.db_path
    output_dir = args.dest

    extract_lmdb(db_path, output_dir)
