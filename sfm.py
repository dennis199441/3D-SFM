import os, cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
from scipy.spatial import Delaunay
from collections import defaultdict
print("Load libraries success")

INPUT_DIR = "./input/"
IMG_SCALE_PERCENT = 60
FEATURE_ALGO = "SIFT"

def getImages(directory):
	files = os.listdir(directory)
	files.sort()
	imgs = []
	for f in files:
		if "jpg" in f.lower() or "png" in f.lower():
			fullPath = directory+ f
			img = cv2.imread(fullPath)
			width = int(img.shape[1] * IMG_SCALE_PERCENT / 100)	
			height = int(img.shape[0] * IMG_SCALE_PERCENT / 100)
			dsize = (width, height)
			output = cv2.resize(img, dsize)
			imgs.append((fullPath, output))
	assert(len(imgs) != 0)
	return imgs

def _getAlgoObjAndIndex(algo):
	FLANN_INDEX_KDTREE = 0
	algoObj, idxParams = None, None
	if algo == "SIFT":
		algoObj = cv2.xfeatures2d.SIFT_create()
		idxParams = dict(algorithm=FLANN_INDEX_KDTREE, tree=5)
	elif algo == "SURF":
		algoObj = cv2.xfeatures2d.SURF_create()
		idxParams = dict(algorithm=FLANN_INDEX_KDTREE, tree=5)
	elif algo == "ORB":
		algoObj = cv2.ORB_create(nfeatures=2000)
		idxParams = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=2)
	else:
		raise Exception(f"Unknown feature detection algorithm: {algo}")
	return algoObj, idxParams

def featureDetection(img, algo="SIFT"):
	grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	flag = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
	algoObj, idxParams = _getAlgoObjAndIndex(algo)	
	kp, des = algoObj.detectAndCompute(grayImg, None)
	kpImg = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=flag)
	cv2.imshow(algo, kpImg)
	cv2.waitKey()

def featureMatching(frames, algo="SIFT"):
	algoObj, idxParams = _getAlgoObjAndIndex(algo)
	# find keypoints and descriptors with SIFT/SURF
	kps, dess = [], []
	for i, img in frames:
		kp, des = algoObj.detectAndCompute(img, None)
		kps.append(kp)
		dess.append(des)

	# FLANN parameters
	searchParams = dict(check=50)
	flann = cv2.FlannBasedMatcher(idxParams, searchParams)
	
	featureMap = defaultdict(lambda: [None] * len(frames))
	ref = dess[0]
	for i in range(1, len(dess)):
		matches = flann.knnMatch(ref, dess[i], k=2)
		for j, (m, n) in enumerate(matches):
			if m.distance < 0.75 * n.distance:
				featureMap[m.queryIdx][0] = kps[0][m.queryIdx].pt
				featureMap[m.queryIdx][i] = kps[i][m.trainIdx].pt
	toRemove = []
	for k, v in featureMap.items():
		if None in v:
			toRemove.append(k)
	for k in toRemove:
		del featureMap[k]
	ptss = [[] for i in range(len(frames))]
	for k, points in featureMap.items():
		for i, point in enumerate(points):
			ptss[i].append(point)
	return ptss	

def _elimate(x, numFeatures):
	sum_u, sum_v = 0, 0
	for u, v in x:
		sum_u += u
		sum_v += v
	u_mean = sum_u / numFeatures
	v_mean = sum_v / numFeatures
	ex = []
	for u, v in x:
		eu = u - u_mean
		ev = v - v_mean
		ex.append((eu, ev))
	return ex

def _buildMeasurementMatrix(exs, numFeatures):
	numFrames = len(exs)
	W = np.zeros((2*numFrames, numFeatures))
	for i, ex in enumerate(exs):
		for j, point in enumerate(ex):
			W[i][j] = point[0]
			W[i + numFrames][j] = point[1]
	return W

def getSingularMatrix(S):
	n = len(S)
	mat = np.zeros((n, n))
	for i, v in enumerate(S):
		mat[i][i] = v
	return mat

def elimateTranslation(xt):
	numFeatures = len(xt[0])
	for x in xt:
		assert(len(x) == numFeatures)
	
	exs = [] # len(exs) = number of frames
	for x in xt:
		ex = _elimate(x, numFeatures)
		exs.append(ex)
	return exs

def factorization(exs, numFeatures):	
	# W is a 2*len(exs) by numFeatures matrix
	print(f"numFeatures: {numFeatures}")
	W = _buildMeasurementMatrix(exs, numFeatures)
	U, S, V = svds(W, k=3) # Note: S is singular VECTOR
	S = getSingularMatrix(S)
	S2 = np.sqrt(S)
	RH = np.matmul(U, S2) # Affine rotation
	SH = np.matmul(S2, V) # Model structure
	return RH, SH

def plot3d(points):
	points = points.T
	fig = plt.figure()
	ax = plt.axes(projection='3d')
	x, y, z = [], [], []
	for point in points:
		x.append(point[0])
		y.append(point[1])
		z.append(point[2])
	ax.scatter3D(x, y, z)
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_zlabel("z")
	plt.show()

def plotDelaunay(points):
	points = points.T
	x, y, z = [], [], []
	for point in points:
		x.append(point[0])
		y.append(point[1])
		z.append(point[2])
	x = np.array(x)
	y = np.array(y)
	z = np.array(z)
	tri = Delaunay(np.array([x,y]).T)
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection='3d')
	ax.plot_trisurf(x, y, z, triangles=tri.simplices)
	plt.show()

def generatePlyFile(points):
	points = points.T
	f = open("output.ply", "w")
	f.write("ply\n")
	f.write("format ascii 1.0\n")
	f.write("comment written by Dennis Cheung\n")
	f.write(f"element vertex {points.shape[0]}\n")
	f.write("property float32 x\n")
	f.write("property float32 y\n")
	f.write("property float32 z\n")
	f.write("property uchar red\n")
	f.write("property uchar green\n")
	f.write("property uchar blue\n")
	f.write("end_header\n")
	for p in points:
		f.write(f"{p[0]} {p[1]} {p[2]} 255 0 0\n")
	f.close()

def testMatch(img1, img2):
	algoObj, idxParams = _getAlgoObjAndIndex(algo=FEATURE_ALGO)
	kp1, des1 = algoObj.detectAndCompute(img1,None)
	kp2, des2 = algoObj.detectAndCompute(img2,None)

	# FLANN parameters
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)   # or pass empty dictionary

	flann = cv2.FlannBasedMatcher(index_params,search_params)
	matches = flann.knnMatch(des1,des2,k=2)
	# Need to draw only good matches, so create a mask
	matchesMask = [[0,0] for i in range(len(matches))]
	# ratio test as per Lowe's paper
	for i,(m,n) in enumerate(matches):
		if m.distance < 0.7*n.distance:
			matchesMask[i]=[1,0]
	draw_params = dict(matchColor = (0,255,0), singlePointColor = (255,0,0), matchesMask = matchesMask,flags = 0)

	img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

	plt.imshow(img3,)
	plt.show()


if __name__ == "__main__":
	frames = getImages(INPUT_DIR)
	# featureDetection(frames[0][1], algo=FEATURE_ALGO) # feature detection example
	# featureDetection(frames[1][1], algo=FEATURE_ALGO)
	# featureDetection(frames[2][1], algo=FEATURE_ALGO)
	# testMatch(frames[0][1], frames[1][1]) # feature matching example
	# testMatch(frames[1][1], frames[2][1])
	# testMatch(frames[0][1], frames[2][1])
	points = featureMatching(frames, algo=FEATURE_ALGO)
	exs = elimateTranslation(points)
	RH, SH = factorization(exs, len(exs[0]))
	# plot3d(SH)	
	# plotDelaunay(SH)
	generatePlyFile(SH)
