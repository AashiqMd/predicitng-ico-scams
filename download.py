import hashlib
import os
import requests
import zipfile


CHECKSUMS = {
    "a8a2f6ebe286697c527eb35a58b5539532e9b3ae3b64d4eb0a46fb657b41562c": "test.txt",
    "ca0b3af855e904b2c9d0ad8d7c48ac680d614cb8feafde882d7a8a72131d352b": "howell_07142019.zip",
    "ef41ea0a67dfdd457289ee8fcf166ee32043a896e152d55e4e59038a66a423f2": "sapkota.zip",
    "2cbbf98dddf590c022a27fadf1ce7d7cb7fb54a7ffadd36d2ce01e1cf9d5160f": "merged_data_rapid_miner.csv",
    "b825c7675ba463735c64b154143e25bcf959f95ad61afc1739e5e809f3091f68": "cryptocurrency_submission_1655932272.txt",
}


def prompt():
    print("What is the ID of the file you would like to download? Ask alexholdenmiller@nyu.edu for it.")
    id = input("> ")
    print(f"Attempting to download [{id}].")
    download(id)

def download(id, checksum=True):
    try:
        os.mkdir("data")
    except FileExistsError:
        pass

    TEMP_DOWNLOAD = os.path.join("data", "download.tmp")
    download_from_google_drive(id, TEMP_DOWNLOAD)

    if checksum:
        sha256_hash = checksum(TEMP_DOWNLOAD)
        if sha256_hash.hexdigest() not in CHECKSUMS:
                raise AssertionError(
                    f"Checksum for {TEMP_DOWNLOAD}\n"
                    f"does not match any expected checksums:\n"
                    f"{sha256_hash.hexdigest()} (received)\n"
                )
        else:
            fn = CHECKSUMS[sha256_hash.hexdigest()]
            os.replace(TEMP_DOWNLOAD, os.path.join("data", fn))
            if fn.endswith(".zip"):
                with open(os.path.join("data", fn), 'rb') as f:
                    with zipfile.ZipFile(f, 'r') as zf:
                        zf.extractall("data/")
                os.remove(os.path.join("data", fn))
            print(f"Download and checksum successful for: {fn}")


### Adapted open source code for downloading from gdrive
### https://github.com/facebookresearch/ParlAI/blob/main/parlai/core/build_data.py
def checksum(path):
    """
    Checksum on a given file.
    :param dpath: path to the downloaded file.
    """
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
        return sha256_hash

def download_from_google_drive(gd_id, destination):
    """
    Use the requests package to download a file from Google Drive.
    """
    URL = 'https://docs.google.com/uc?export=download'

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    with requests.Session() as session:
        response = session.get(URL, params={'id': gd_id}, stream=True)
        token = get_confirm_token(response) or 't'

        if token:
            response.close()
            params = {'id': gd_id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)

        CHUNK_SIZE = 32768
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
        response.close()
###

if __name__ == "__main__":
    download()