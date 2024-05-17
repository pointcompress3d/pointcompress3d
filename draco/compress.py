import os
import DracoPy
import open3d as o3d
import sys
import time
import numpy as np
# Note: If mesh.points is an integer numpy array,
# it will be encoded as an integer attribute. Otherwise,
# it will be encoded as floating point.



times = []
out_folder = f"{sys.argv[2]}{sys.argv[3]}_{sys.argv[4]}/"
print(out_folder)
for root, dirs, files in os.walk(sys.argv[1]):
    for file in files:
        # Check if the file has a PCD extension
        if file.endswith(".pcd"):
            file_path = os.path.join(root, file)
            # Process the PCD file
            pc = o3d.io.read_point_cloud(file_path)
            # Options for encoding:
            start_time = time.time()
            binary = DracoPy.encode(
                pc.points,
                quantization_bits=int(sys.argv[3]), compression_level=int(sys.argv[4]),
                quantization_range=-1, quantization_origin=None,
                create_metadata=False, preserve_order=False,
            )
            end_time = time.time()
            times.append(end_time - start_time)
            store_path = f"{out_folder}{os.path.basename(file_path).split('.')[0]}.drc"
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            with open(store_path, 'wb') as f:
                f.write(binary)

print("Encoding Mean Time:", np.mean(np.array(times)))