# About the files
1. The BEV and detection adn point cloud visualization is doen using main.py.
2. There is another file named sam_seg.py which properly uses Segment anything model SAM.and then uses yolo to detct cars hence provoding us with a very precisie and good semantically segmented cars.
3. Only bleedout filtering is also done using sam_seg.py.
4. Yolov8 weights have been also uploaded and are required in sam_seg.py meanwhile main.py uses yolov11-seg weights. 
5. There are several other code which still have to be modifi3ed as that wuld be the ROS nodes for us.
## Previous 3d data without bleedout filtering
![image](https://github.com/user-attachments/assets/94182c7f-0628-4b05-89cf-6548ddf1bdd6)
## Current 3d data with bleedout filtering 
![image](https://github.com/user-attachments/assets/4732232a-06a0-4e8d-a661-7d9ac8b1845b)

## Current BEV (from main.py)
![image](https://github.com/user-attachments/assets/72f9a39f-947f-4009-b56a-1d207485e7f6)

## Semantic Segmentation (using yolo [main.py])
![image](https://github.com/user-attachments/assets/c59e3469-2e34-451a-8731-0d4f351b77dd)
## Semantic Segmentation (SAM)
![image](https://github.com/user-attachments/assets/30f04fef-8cf0-4173-81b3-624a26a61ffb)

