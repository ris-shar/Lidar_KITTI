from setuptools import setup
from glob import glob
import os

package_name = 'kitti_project'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        # Python entry-points live in resource index
        (os.path.join('share', 'ament_index', 'resource_index', 'packages'),
         [os.path.join('resource', package_name)]),
        # launch files
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.py')),
        # package.xml
        (os.path.join('share', package_name), ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'opencv-python',
        'open3d',
        'scikit-learn',
        'ultralytics',
        'matplotlib',
    ],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='Semantic segmentation, point-cloud and BEV nodes for KITTI',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'semantic_node = kitti_project.semantic_segmenter:main',
            'pc_node        = kitti_project.pointcloud_processor:main',
            'bev_node       = kitti_project.bev_generator:main',
        ],
    },
)
