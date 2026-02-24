# Get data
import os
from dotenv import load_dotenv
from roboflow import Roboflow

# Load environment variables from .env file
load_dotenv()


# Download datasets

# Dataset 1 - Open Shelves
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("capjamesg").project("open-shelves")
dataset = project.version(9).download("yolov8")
# 163 images

# @misc{
#                             open-shelves_dataset,
#                             title = { Open Shelves Dataset },
#                             type = { Open Source Dataset },
#                             author = { capjamesg },
#                             howpublished = { \url{ https://universe.roboflow.com/capjamesg/open-shelves } },
#                             url = { https://universe.roboflow.com/capjamesg/open-shelves },
#                             journal = { Roboflow Universe },
#                             publisher = { Roboflow },
#                             year = { 2025 },
#                             month = { oct },
#                             note = { visited on 2026-02-23 },
#                             }

# Dataset 2 - Libvision Dataset
project = rf.workspace("libvision").project("libvision")
version = project.version(2)
dataset = version.download("yolov8")
# 961 images

# @misc{
#                             libvision_dataset,
#                             title = { LibVision Dataset },
#                             type = { Open Source Dataset },
#                             author = { LibVision },
#                             howpublished = { \url{ https://universe.roboflow.com/libvision/libvision } },
#                             url = { https://universe.roboflow.com/libvision/libvision },
#                             journal = { Roboflow Universe },
#                             publisher = { Roboflow },
#                             year = { 2024 },
#                             month = { oct },
#                             note = { visited on 2026-02-24 },
#                             }
                

