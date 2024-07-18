import os
import DracoPy
import open3d as o3d
import sys
import time
import numpy as np
# Note: If mesh.points is an integer numpy array,
# it will be encoded as an integer attribute. Otherwise,
# it will be encoded as floating point.



def construct_path(folder):
    # Check if the provided folder path is absolute
    if not os.path.isabs(folder):
        # If it's not absolute, combine it with the current working directory
        folder = os.path.join(os.getcwd(), folder)
    return folder

def remove_empty_elements(arr):
    return list(filter(None, arr))


times = []
param_folder = remove_empty_elements(sys.argv[1].split('/'))[-1]

for root, dirs, files in os.walk(sys.argv[1]):
    for file in files:
        # Check if the file has a .drc extension
        if file.endswith(".drc"):
            file_path = os.path.join(root, file)
            with open(file_path, 'rb') as draco_file:
                start_time = time.time()
                mesh = DracoPy.decode(draco_file.read())
                end_time = time.time()
                times.append(end_time - start_time)
                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(mesh.points)
                 
                file_output = os.path.join(os.getcwd(), sys.argv[2], param_folder)
                if not os.path.exists(file_output):
                    os.makedirs(file_output) 

                filename = os.path.splitext(file)[0]
                file_output = os.path.join(file_output, filename) + '.pcd'
                o3d.io.write_point_cloud(file_output, pc)

print("Decoding Mean Time:", np.mean(np.array(times)))
