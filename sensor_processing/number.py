import numpy as np
import math
import pandas as pd
import os

def main():
    table = pd.read_csv("Pandar128_Angle_Correction_File.csv", encoding="utf-8", on_bad_lines="skip")
    destination_folder_path = "/home/ailab/AILabDataset/01_Open_Dataset/39_AutoDna/3d_object_detection/3d_mod_av_db/Team_1_128to64"
    files = os.listdir(destination_folder_path)
    print(len(files))

if __name__ == "__main__":
    main()