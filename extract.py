import zipfile


def extract(filename):
    print('Extracting {}...'.format(filename))
    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall('data')
    zip_ref.close()


if __name__ == "__main__":
    extract('data/Challenge2_Training_Task12_Images.zip')
    extract('data/Challenge2_Training_Task1_GT.zip')
    extract('data/ch4_training_images.zip')
    extract('data/ch4_training_localization_transcription_gt.zip')
