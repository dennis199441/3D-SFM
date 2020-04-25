import os, cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
from collections import defaultdict
print("Load libraries success")

INPUT_DIR = "./input/"
IMG_SCALE_PERCENT = 20
FEATURE_ALGO = "SURF"

def getImages(directory):
	files = os.listdir(directory)
	files.sort()
	imgs = []
	for f in files:
		if "jpg" in f.lower() or "png" in f.lower():
			fullPath = directory+ f
			print(f"=== [TEST] Loading {fullPath}...")
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
	print(f"=== [TEST] featureMap: number of feature = {len(featureMap)}")
	ptss = [[] for i in range(len(frames))]
	for k, points in featureMap.items():
		for i, point in enumerate(points):
			ptss[i].append(point)
	return ptss	

def _elimateTranslation(x, numFeatures):
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

def subFactorizationLinear(Mhat, F):
	pass	

def factorization(xt):
	numFeatures = len(xt[0])
	for x in xt:
		assert(len(x) == numFeatures)
	
	exs = [] # len(exs) = number of frames
	for x in xt:
		ex = _elimateTranslation(x, numFeatures)
		exs.append(ex)
	
	# W is a 2*len(exs) by numFeatures matrix
	W = _buildMeasurementMatrix(exs, numFeatures)
	print(f"=== [TEST] _buildMeasurementMatrix: W.shape = {W.shape}")
	U, S, V = svds(W, k=3) # Note: S is singular VECTOR
	print(f"=== [TEST] svds U.shape = {U.shape}; S.shape = {S.shape}; V.shape = {V.shape}")
	S = getSingularMatrix(S)
	S2 = np.sqrt(S)
	RH = np.matmul(U, S2) # Affine rotation
	SH = np.matmul(S2, V) # Model structure
	#Q = subFactorizationLinear(SH, 2*len(exs)))i
	#R = np.matmul(RH, Q)
	#S = np.matmul(np.linalg.pinv(Q), SH)
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

if __name__ == "__main__":
	frames = getImages(INPUT_DIR)
	print(f"=== [TEST] {len(frames)} frames are found...")

	for p, f in frames:
		print(f"=== [TEST] featureDetection: {p} [START]")
		featureDetection(f, algo=FEATURE_ALGO)

	print("=== [TEST] featureMatching")
	ptss = featureMatching(frames, algo=FEATURE_ALGO)

	print(f"=== [TEST] factorization: {len(ptss)} frames, {len(ptss[0])} features")
	RH, SH = factorization(ptss)
	print(f"=== [TEST] RH.shape = {RH.shape}; SH.shape = {SH.shape}")
	
	plot3d(SH)	
