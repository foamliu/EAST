import zipfile

from config import training_data_path, test_data_path
from utils import ensure_folder


def extract(filename, folder):
    print('Extracting {}...'.format(filename))
    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall(folder)
    zip_ref.close()


if __name__ == "__main__":
    ensure_folder(training_data_path)
    extract('data/ch4_training_images.zip', training_data_path)
    extract('data/ch4_training_localization_transcription_gt.zip', training_data_path)

    ensure_folder(test_data_path)
    extract('data/ch4_test_images.zip', test_data_path)
    extract('data/Challenge4_Test_Task1_GT.zip', test_data_path)
