import os.path
import sys

import h5py
import numpy as np
import open3d as o3d


# This function is only for resampling purpose from raw dyna dataset.
# Once you've downloaded the resampled dyna dataset, it is useless.
def _resample_dyna():
    f1 = h5py.File(
        "datasets/dyna_male.h5", "r"
    )

    f2 = h5py.File(
        "datasets/dyna_female.h5", "r"
    )

    wf = h5py.File(
        "datasets/dyna_poisson_8192_resampled.h5", "w"
    )

    def _resample(f: h5py.File, w: h5py.File):
        faces = f["faces"][()]

        for item in f.keys():
            if item == "faces":
                continue

            vertices = f1[item][()].transpose([2, 0, 1])
            resampled_vertices = []
            for i in range(len(vertices)):
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(vertices[i])
                mesh.triangles = o3d.utility.Vector3iVector(faces)

                pcd = mesh.sample_points_poisson_disk(number_of_points=8192, init_factor=1)
                resampled_vertices.append(np.asarray(pcd.points))

                print("\r", end="")
                print(f"Resample for item: {item}, progress: {i + 1}/{len(vertices)}", end="")
            resampled_vertices = np.asarray(resampled_vertices)

            w.create_dataset(item, data=resampled_vertices)
            print("")

    _resample(f1, wf)
    _resample(f2, wf)
    wf.close()


_resample_dyna()
