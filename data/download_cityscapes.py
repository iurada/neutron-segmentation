import os
from cityscapesscripts.download import downloader as dw

DESTINATION_PATH = 'data/Cityscapes'

if __name__ == '__main__':
    os.makedirs(DESTINATION_PATH, exist_ok=True)

    s = dw.login()
    dw.download_packages(session=s, 
                         package_names=['gtFine_trainvaltest.zip', 'leftImg8bit_trainvaltest.zip'], 
                         destination_path=DESTINATION_PATH)