## Development environment
1. macOS Catalina 10.15.3
2. Python 3.7.3

## Dependencies
1. opencv-python==3.4.2.16
2. (Important) opencv-contrib-python==3.4.2.16
3. numpy==1.18.3
4. matplotlib==3.2.1
5. plyfile==0.7.2
6. PyQt5==5.14.2
7. mayavi==4.7.1

## How to run
1. Create virtual environment using the following command:
	```
	virtualenv env
	```
2. Activate the virtual environment using the following command:
	```
	source env/bin/activate
	```
3. Install dependencies using the following command:
	```
	pip3 install -r requirements.txt
	```
4. Put your image into input folder

5. Execute the following commands:
	```
	python3 sfm.py

	python3 visual.py
	```

## Notes
1. Must use Python 3.7 or above as we use f-strings
2. opencv-python and opencv-contrib-python must be version 3.4.2.16
3. If you have trouble during the installation process of mayavi and PyQt5, you can ignore these libraries as they are used for visualizing the PLY file. You can use other software such as meshlab to visualize the PLY file.