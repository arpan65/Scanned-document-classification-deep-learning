# Code to unzip the raw data
file_path="rvl-cdip.tar.gz"
import tarfile
tar = tarfile.open(file_path)
tar.extractall()
tar.close()