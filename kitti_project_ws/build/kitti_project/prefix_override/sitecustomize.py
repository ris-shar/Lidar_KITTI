import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/rishav/PycharmProjects/PythonProject/kitti_project_ws/install/kitti_project'
