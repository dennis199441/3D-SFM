import numpy as np
from mayavi import mlab
from plyfile import PlyData
print("Loaded libraries")

FILENAME = "./output.ply"

def main():
	mlab.figure(bgcolor=(0,0,0))
	plot(PlyData.read(FILENAME))
	mlab.show()

def plot(ply):
	print(f"Start to plot {FILENAME}")
	vertex = ply['vertex']
	(x, y, z) = (vertex[t] for t in ('x', 'y', 'z'))
	mlab.points3d(x, y, z, color=(1,1,1), mode='sphere', scale_factor=0.5, resolution=100)
	if 'face' in ply:
		tri_idx = ply['face']['vertex_indices']
		triangles = np.vstack(tri_idx)
		mlab.triangular_mesh(x, y, z, triangles, color=(1,0,0.4), opacity=0.5)

if __name__ == "__main__":
	main()
