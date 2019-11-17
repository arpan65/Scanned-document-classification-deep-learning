# Scanned-Document-Classification
BFSI sectors deal with lots of unstructured scanned documents which are archived in document management systems for further use.For example in Insurance sector, when a policy goes for underwriting, underwriters attached several raw notes with the policy, Insureds also attach various kind of scanned documents like identity card, bank statement, letters etc. In later parts of the policy life cycle if claims are made on a policy, releted scanned documents also archeived.Now it becomes a tedious job to identify a particular document from this vast repository. The goal of this case study is to develop a deep learning based solution which can automatically classify scanned documents.

# Data
We will use the RVL-CDIP (Ryerson Vision Lab Complex Document Information Processing) dataset which consists of 400,000 grayscale images in 16 classes, with 25,000 images per class. There are 320,000 training images, 40,000 validation images, and 40,000 test images. The images are sized so their largest dimension does not exceed 1000 pixels.

link : https://www.cs.cmu.edu/~aharley/rvl-cdip/

# Folder Structure
Data -- link to download data
Models -- link to download trained models
Notebooks --DocManagement.ipynb (Anchor notebook), paramtune (Notebook for paramenter tunning)
PDF -- Notebook snapshots in PDF format
Scripts -- Necessary scripts to recreate the result

# License
