import pandas as pd
import os
import glob


# TODO to change dayTraining-dayClip... to dayTraining-dayClip1


def add_headers_tod_csv(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = ['ImageID', 'Xmin', 'Ymin', 'Xmax', 'Ymax', 'ClassName']
    df.to_csv(csv_path, index=False)
    return df
def change_filename_data():
    path1 = "/home/zeys/Downloads/archive/ISUZU_BUSODD/lisa_annotations/sample-dayClip6.csv"
    path2 = "/home/zeys/Downloads/archive/ISUZU_BUSODD/lisa_annotations/sample-nightClip1.csv"
    df1 = pd.read_csv(path1, sep=';')
    df2 = pd.read_csv(path2, sep=';')
    df1['Filename'] = df1['Filename'].str.replace('/', '-')
    df1.to_csv("/home/zeys/Downloads/archive/ISUZU_BUSODD/lisa_annotations/sample-dayClip6.csv", index=False)
    df2['Filename'] = df2['Filename'].str.replace('/', '-')
    df2.to_csv("/home/zeys/Downloads/archive/ISUZU_BUSODD/lisa_annotations/sample-nightClip1.csv", index=False)


# try to change my own dataset headers  similar to open images dataset headers
def change_headers():
    path = "/home/zeys/Downloads/archive/ISUZU_BUSODD/lisa_annotations/deneme/"
    csv_files_path = glob.glob(os.path.join(path, "*.csv"))
    for csv_file in csv_files_path:
        df = pd.read_csv(csv_file)
        df = df.rename(columns={'Filename': 'ImageID', 'Annotation Tag ': 'ClassName', 'Upper left corner X': 'Xmin',
                                'Upper left corner Y': 'Ymin', 'Lower right corner X': 'Xmax',
                                'Lower right corner Y': 'Ymax'})
        df['ClassName'] = 'traffic_light'
        df.to_csv(csv_file, index=False)

def concanate_csv_files():
    path = "/media/zeys/PortableSSD/pre_csv_files/"
    csv_files_path = glob.glob(os.path.join(path, "*.csv"))
    # for loop to read all csv files
    df = pd.concat([pd.read_csv(f) for f in csv_files_path])
    df.to_csv("/media/zeys/PortableSSD/pre_csv_files/total_dataset.csv", index=False)

# change_headers()
# change_filename_data()
concanate_csv_files()

