# get_dataset.py

# Example usage:
# python get_dataset.py --help
# python get_dataset.py -d data -rd n
# python get_dataset.py -d data -rd y -rt y
# python get_dataset.py -d data -rd n -ruc n -s https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip
# python get_dataset.py -d data -rd n -ruc n -s https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_LesionGroupings.csv
# python get_dataset.py -d data -rd n -ruc n -s https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_GroundTruth.zip
# python get_dataset.py -d data -rd n -ruc n -s https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Validation_Input.zip
# python get_dataset.py -d data -rd n -ruc n -s https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Validation_GroundTruth.zip

import argparse
from src.download_helper import DownloadHelper

DOWNLOAD_URL = 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_LesionGroupings.csv'


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description='Download the target training dataset')
    parser.add_argument('--data_dir', '-d', type=str, help="Path to the root target data director")
    parser.add_argument('--replace_download', '-rd', type=str, default='n',
                        help="Flag to overwrite existing download file")
    parser.add_argument('--replace_unzip_content', '-ruc', type=str, default='n',
                        help="Flag to replace existing training folder content")
    parser.add_argument('--src_url', '-s', type=str, default=DOWNLOAD_URL,
                        help="Source URL for download")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    DownloadHelper.download_dataset(**args.__dict__)
