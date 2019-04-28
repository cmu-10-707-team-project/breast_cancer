from __future__ import print_function

import argparse
import os.path
import zipfile
from os import path

import pandas as pd

from data_process.google_drive_utils import create_drive_service, \
    drive_download_one_file

GOOGLE_DRIVE_FILE_LINK = 'https://drive.google.com/file/d/{}/view'

FILE_LIST_PAGE_SIZE = 500

CAMELYON16_FOLDER_ID = '0BzsdkU4jWx9Bb19WNndQTlUwb2M'


def _drive_list_files(service, folder_id):
    results = service.files().list(
        q=("'%s' in parents" % folder_id),
        pageSize=FILE_LIST_PAGE_SIZE, fields="nextPageToken, files(id, name)").execute()
    return results.get('files', [])


def _extract_file_ids(items, file_names):
    file_id_lookup = {}

    for f in items:
        if f['name'] in file_names:
            file_id_lookup[f['name']] = f['id']

    return file_id_lookup


def _download_and_unzip_annotations(service, out_path, ann_id, dryrun=False):
    ann_zip_path = out_path

    drive_download_one_file(
        service, ann_zip_path, ann_id, dryrun=dryrun)

    if not dryrun:
        # unzip the annotations
        ann_unzipped_path = out_path.rstrip('.zip')
        zip_ref = zipfile.ZipFile(ann_zip_path, 'r')
        zip_ref.extractall(ann_unzipped_path)
        zip_ref.close()

    os.remove(ann_zip_path)


def _download_train_set(service, out_folder_path, train_folder_id, dryrun=False):
    items = _drive_list_files(service, train_folder_id)

    folders = _extract_file_ids(items, ['normal', 'tumor', 'lesion_annotations.zip'])
    normal_images = _drive_list_files(service, folders['normal'])
    tumor_images = _drive_list_files(service, folders['tumor'])

    _download_and_unzip_annotations(
        service,
        path.join(out_folder_path, 'train_annotations.zip'),
        folders['lesion_annotations.zip'],
        dryrun=False
    )

    index = []

    for img in normal_images:
        downloaded_name = '{}_{}'.format('train', img['name'])
        drive_download_one_file(
            service,
            path.join(out_folder_path, downloaded_name),
            img['id'],
            dryrun=dryrun
        )
        ann_name = img['name'].replace('.tif', '.xml')
        index.append(
            {'image_file': downloaded_name,
             'id': img['name'].rstrip('.tif'),
             'google_drive_fileid': img['id'],
             'google_drive_link': GOOGLE_DRIVE_FILE_LINK.format(img['id']),
             'label': 'normal',
             'annotation_file': path.join('train_annotations', ann_name)
             })

    for img in tumor_images:
        downloaded_name = '{}_{}'.format('train', img['name'])
        drive_download_one_file(
            service,
            path.join(out_folder_path, downloaded_name),
            img['id'],
            dryrun=dryrun
        )
        ann_name = img['name'].replace('.tif', '.xml')
        index.append(
            {'image_file': downloaded_name,
             'id': img['name'].rstrip('.tif'),
             'google_drive_fileid': img['id'],
             'google_drive_link': GOOGLE_DRIVE_FILE_LINK.format(img['id']),
             'label': 'tumor',
             'annotation_file': path.join('train_annotations', ann_name)
             })

    index_df = pd.DataFrame(index).sort_values('id')
    index_df.to_csv(path.join(out_folder_path, 'index_train.csv'), index=None)


def _download_test_set(service, out_folder_path, test_folder_id, dryrun=False):
    items = _drive_list_files(service, test_folder_id)

    folders = _extract_file_ids(items, ['images', 'lesion_annotations.zip', 'reference.csv'])
    images = _drive_list_files(service, folders['images'])

    _download_and_unzip_annotations(
        service,
        path.join(out_folder_path, 'test_annotations.zip'),
        folders['lesion_annotations.zip'],
        dryrun=False
    )

    drive_download_one_file(
        service,
        path.join(out_folder_path, '_reference.csv'),
        folders['reference.csv'],
        dryrun=False
    )

    index = []
    for img in images:
        drive_download_one_file(
            service,
            path.join(out_folder_path, img['name']),
            img['id'],
            dryrun=dryrun
        )
        ann_name = img['name'].replace('.tif', '.xml')
        index.append(
            {'image_file': img['name'],
             'id': img['name'].rstrip('.tif'),
             'google_drive_fileid': img['id'],
             'google_drive_link': GOOGLE_DRIVE_FILE_LINK.format(img['id']),
             'annotation_file': path.join('test_annotations', ann_name)
             })

    index_df = pd.DataFrame(index).set_index('id')
    ref_df = pd.read_csv(
        path.join(out_folder_path, '_reference.csv'),
        index_col='id',
        names=['id', 'label', 'type', 'size'])
    index_df = index_df.merge(ref_df, left_index=True, right_index=True).sort_index()
    index_df.to_csv(path.join(out_folder_path, 'index_test.csv'), index_label='id')

    os.remove(path.join(out_folder_path, '_reference.csv'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-folder', default=str, required=True)
    parser.add_argument('--test-folder', default=str, required=True)
    parser.add_argument('--dry-run', action='store_true')
    arg = parser.parse_args()

    if not arg.dry_run:
        os.makedirs(arg.train_folder, exist_ok=True)
        os.makedirs(arg.test_folder, exist_ok=True)

    service = create_drive_service()

    # find train and test folder
    items = _drive_list_files(service, CAMELYON16_FOLDER_ID)
    folders = _extract_file_ids(items, ['training', 'testing'])

    _download_train_set(
        service, arg.train_folder, folders['training'], dryrun=arg.dry_run)
    _download_test_set(
        service, arg.test_folder, folders['testing'], dryrun=arg.dry_run)


if __name__ == '__main__':
    main()
