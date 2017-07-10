import cv2
from imutils import paths
import numpy as np
import argparse
import glob

sudoku_raw='Sudoku/'
sudoku_input='input/'
output_solved='output/'

def auto_canny(image, sigma=0.33):
	image=cv2.GaussianBlur(image, (3, 3), 0)
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged

def main():
	sudokuPaths=glob.glob(sudoku_input+"*.jpg")
	#print sudokuPaths
	j=0
	#for sudokuPath in sudokuPaths:
	sudokuPath = "input/in_0.jpg"
	bgr=cv2.imread(sudokuPath)
	img=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
	img2=np.zeros((1041,700), dtype=np.uint8)
	img2[:]=255
	img2[:,7:693]=img


	ret,img2 = cv2.threshold(img2,127,255,cv2.THRESH_BINARY)
	img3=img2.copy()
#	img3=img2.copy()
	
	#cv2.imshow('thresh',img2)
	temp,hierarchy = cv2.findContours(img2, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#	img3=auto_canny(img3)
#	img4=img3.copy()
#	cv2.imshow('img3',img3)
	# Vertical lines
#	lines = cv2.HoughLinesP(
#		img3, 1, np.pi/180, threshold=100, minLineLength=30, maxLineGap=1)[0]

#	for x1,y1,x2,y2 in lines:        
#		cv2.line(img,(x1,y1),(x2,y2),(255,255,255),3)
#	for x1,y1,x2,y2 in lines2:        
#		cv2.line(img,(x1,y1),(x2,y2),(255,255,255),3)

	contours2 = []
	rect_corner=[]
	roi=[]
	for i in temp:
		if cv2.contourArea(i)>100000 and cv2.contourArea(i)<110000:
			contours2.append(i)
			#print cv2.contourArea(i)
	for c in contours2:
		rect = cv2.boundingRect(c)
		y,x,w,h = rect
		rect_corner.append([x,y,w])
		roi=img3[x+1:x+w-1,y+1:y+h-1]
		print roi.shape[:2]
		imgPath='input/in_'+str(j)+'.jpg'
		cv2.imwrite(imgPath,roi)
		j+=1
		print (sudokuPath,imgPath)
		cv2.imshow('roi',roi)
		if cv2.waitKey(32) & 0xFF==ord('q'):
			cv2.destroyAllWindows()
			break

	'''
	i=0
	while(i<6):
		y,x,w=rect_corner[i]
		#print x,y
		#print img.shape[:2]
		
		l=1
		while(l<10):
			j=1
			while(j<10):
				roi.append(img[x+4+(w/9)*(l-1):x-1+(w/9)*l,y+4+(w/9)*(j-1):y-4+(w/9)*j])
				cv2.imshow('roi',roi[81*i+9*(l-1)+j-1])
				cv2.imwrite('digits/dig_'+str(len(roi))+'.jpg',roi[81*i+9*(l-1)+j-1])
				j+=1
				k=cv2.waitKey(32)
				if k & 0xFF==ord('q'):
					cv2.destroyAllWindows()
			l+=1
		i+=1
	print len(roi)
#cv2.imshow('roi',roi)

cv2.imshow('test', img)
k=cv2.waitKey(0)
if k & 0xFF==ord('q'):
	cv2.destroyAllWindows()


	constants----
	Area of sudoku = 100487
	side of each sudoku = 318x318
	so each small square = (w/9) edge

	'''
if __name__ == '__main__':
	main()