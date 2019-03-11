# Deep Learning for Metastatic Breast CancerDetection

## Setup
Install dependencies.
```
pip install -r requirements.txt
```

## Data Preprocessing

### Download
`cd` into `data_process`. Make sure `credential.json` exits in `data_process`. With the `--dry-run` flag,
the script will not download the slide images, but the annotations will still be downloaded.
```
python --train-folder TRAIN_FOLDER --test-folder TEST_FOLDER [--dry-run]
```
