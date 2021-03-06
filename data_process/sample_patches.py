import multiresolutionimageinterface as mir
from collections import Counter
from os import path

from skimage import color, io

import numpy as np
import pandas as pd
from tqdm import tqdm

from deephistopath.wsi import filter

PATCH_HEIGHT = 256
PATCH_WIDTH = 256

DS_LEVEL0 = 0
DS_LEVEL8 = 8


def parse_one_annotation_list(reader, slide_path, ann_path, mask_path):
    slide = reader.open(slide_path)
    annotation_list = mir.AnnotationList()
    xml_repository = mir.XmlRepository(annotation_list)
    xml_repository.setSource(ann_path)
    xml_repository.load()
    annotation_mask = mir.AnnotationToMask()
    label_map = {'_0': 1, '_1': 1, '_2': 0}
    conversion_order = ['_0', '_1', '_2']
    annotation_mask.convert(
        annotation_list, mask_path, slide.getDimensions(),
        slide.getSpacing(), label_map, conversion_order)
    slide.close()


def sample_one_slide_image(
        reader, slide_id, slide_path, mask_path, output, pos_to_neg_ratio=1.,
        neg_sample_rate=1., dryrun=False):
    slide = reader.open(slide_path)

    x_l0, y_l0 = slide.getLevelDimensions(DS_LEVEL0)
    x_l8, y_l8 = (x_l0 // PATCH_WIDTH, y_l0 // PATCH_HEIGHT)

    slide_l8 = slide.getUCharPatch(0, 0, x_l8, y_l8, DS_LEVEL8)
    slide_l8_filtered = color.rgb2gray(filter.apply_image_filters(slide_l8))

    if mask_path is not None:
        # first pass through slide image and count # of positive/negative samples
        mask = reader.open(mask_path)
        c = Counter()
        for y_i in range(0, y_l8):
            for x_i in range(0, x_l8):
                if slide_l8_filtered[y_i, x_i] == 0:
                    continue

                # the pixel is tissue
                mask_patch = mask.getUCharPatch(
                    x_i * PATCH_WIDTH, y_i * PATCH_HEIGHT,
                    PATCH_WIDTH, PATCH_HEIGHT, DS_LEVEL0)

                if np.sum(mask_patch) > 0:
                    c['positive'] += 1
                else:
                    c['negative'] += 1

        if pos_to_neg_ratio is not None:
            neg_sample_rate = c['positive'] / pos_to_neg_ratio / c['negative']
        else:
            neg_sample_rate = 1.

        print('start sampling slide {}, {} positive samples and {} negative '
              'samples in total, negative sampling rate is {}'.format(
            slide_id, c['positive'], c['negative'], neg_sample_rate))
    else:
        print('start sampling slide {} negative sampling rate is {}'.format(
            slide_id, neg_sample_rate))

    if dryrun:
        return

    # second pass through slide image and sample
    index = []
    for y_i in tqdm(range(0, y_l8)):
        for x_i in range(0, x_l8):
            if slide_l8_filtered[y_i, x_i] == 0:
                continue

            if mask_path is not None:
                # the pixel is tissue
                mask_patch = mask.getUCharPatch(
                    x_i * PATCH_WIDTH, y_i * PATCH_HEIGHT, PATCH_WIDTH,
                    PATCH_HEIGHT, DS_LEVEL0)

                tumor_prob = np.sum(mask_patch) / (PATCH_HEIGHT * PATCH_WIDTH)
            else:
                tumor_prob = 0

            if tumor_prob == 0:
                # sample negative patches according to the sampling rate
                if np.random.uniform(0, 1) > neg_sample_rate:
                    continue

            slide_patch = slide.getUCharPatch(
                x_i * PATCH_WIDTH, y_i * PATCH_HEIGHT, PATCH_WIDTH,
                PATCH_HEIGHT, DS_LEVEL0)

            patch_id = '{}_{}_{}'.format(slide_id, x_i, y_i)
            patch_filename = '{}.png'.format(patch_id)

            if tumor_prob == 0:
                # construct an empty mask
                mask_patch = np.zeros(
                    list(slide_patch.shape[0:-1]) + [1], dtype=np.int)

            stacked = np.concatenate([slide_patch, mask_patch], axis=2)
            io.imsave(path.join(output, patch_filename), stacked)

            index.append(
                {
                    'filename': patch_filename,
                    'tumor_prob': tumor_prob,
                    'patch_id': patch_id,
                    'slide_id': slide_id
                 })

    slide.close()
    if mask_path is not None:
        mask.close()

    return pd.DataFrame(index).set_index('patch_id')
