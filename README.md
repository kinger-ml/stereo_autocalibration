Project Title: Stereo Online-Autocalibration on Mobile Hardware
Authors: Krishna Kinger
Contact: krishnakinger@gmail.com

Prerequisite:
Python 3.6.9
OPENCV 3.3.0
CVSBA: https://www.uco.es/investiga/grupos/ava/node/39

Project Structure:
-> cpp directory includes bundle adjustment code. It includes cmake
to build the output for bundle adjustment. Paste the Executable in python directory.
-> python directory includes the feature detection, stereo matching
and temporal feature matching.

For six-way matching, use main.py file and bundle adjustment executable
built using sba2d.cpp

For four-way matching, use main_lite.py and bundle adjustment executable
built using sba2d_lite.cpp
Paste the executable to python directory.

Dataset used: https://s3.eu-central-1.amazonaws.com/avg-karlsruhe/2010_03_17_drive_0046.zip
Paste the dataset in python/data/2010_03_17_drive_0046
For other dataset camera parameters must be set in python/camera_params.py

The program requires the initial camera extrinsics to be declared in file. For the above dataset,
the files are saved in python/data/camera/ directory
