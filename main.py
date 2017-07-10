import cv2
from imutils import paths
import numpy as np
from matplotlib import pyplot as plt
from random import randint
import math
from Tkinter import Tk
from tkFileDialog import askopenfilename

'''
sudoku_raw='Sudoku/'
sudoku_input='input/'
output_solved='output/'
'''

def main():

	Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
	filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
	#print(filename)

	sudokuPath=filename
	#print sudokuPath
	bgr=cv2.imread(sudokuPath)
	img=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)

	img2=img.copy()

	w=316
	#to create and initialize a 2D array named ROI 
	roi   = [[img2 for i in range(9)] for j in range(9)]
	matrix = [[0 for i in range(9)] for j in range(9)]
	x=0
	y=0
	l=0
	pred=0

	while(l<9):
		j=0
		while(j<9):
			roi[l][j]=img[x+2+(w/9)*(l):x-2+(w/9)*(l+1),y+2+(w/9)*(j):y-2+(w/9)*(j+1)]

			#Now, we try to match template and predict the number
			#cv2.imwrite('digits/dig_'+str(len(roi))+'.jpg',roi[81*i+9*(l-1)+j-1])
			pred = match(roi[l][j])
			matrix[l][j] = pred

			
			'''
			cv2.namedWindow(str(pred),cv2.WINDOW_NORMAL)
			cv2.imshow(str(pred),roi[l][j])
			k=cv2.waitKey(0)
			if k & 0xFF==ord('n'):
				j+=1
				cv2.destroyWindow(str(pred))
				continue
			elif k & 0xFF==ord('q'):
				cv2.destroyAllWindows()
			'''
			j+=1
		l+=1
	print matrix

	#now the actual algo will work
	'''
			Algo:
				1. Conditions:
							a. Each 3x3 cell must contain numbers from 1 to 9
							b. Each row and column must contain numbers from 1 to 9
				2. How to do it:
							a. Create a temporary array containing the probables for a particular cell, i.e., the numbers which that cell allows to be filled in subject to Conditions
							b. Method - Brute Force --
									i. Here we try all the possible combinations until we get our answer
									ii. Simplistic Implementation:
													-> Create possible combinations for each 3x3 cell, satisfying 1 and using a.
													-> Choose one combination for each 3x3 cell and check for 1 again in each filled cell
													-> If any filled cell fails condition, store this matrix in failed matrices set and repeat
									iii. Smart Implementation:
													-> Method: Make a list of all the to-be-filled cells.
															-- Select one cell at random and check if it allows just one number to be filled
															-- If it does, fill the number, update the list of to-be-filled cells, then back to first step
	'''

	#an infinite while loop with exit condition being either completing the sudoku or reaching a max number of tries
	#create a list of cells which contain 0 in matrix
	
	listofcells = makeList(matrix)

	completed = False
	tries=0
	tempNum = []
	for i in range(1,10):
		tempNum.append(i)

	
	#NOTE: listofcells[x][y] :: Here, x is the horizontal column number, y is the vertical row number
	centres = []
	for i in range(3):
		for j in range(3):
			centres.append([i*3+1,j*3+1])
	print centres

	img2=cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)

	while(completed!=True and tries<1000):
		temp = list(tempNum)
		tries+=1
		numofcells = len(listofcells)
		indexofcell = randint(0,(numofcells-1))
		#print (numofcells,indexofcell)
		cell = listofcells[indexofcell]
		print "cell: %s" %(cell)


		# now check this cell for conditions
		# if any condition is false, continue
		
		# 1. 3x3 cell check
		# Algo: Calc the distance of the cell wrt all the 3x3 block centres and hence find the block in which the cell resides
		#		Then, populate all the cells in that block and pass them to remove
		block = findBlock(cell, centres)
		centre = centres[block]
		for i in range(-1,2):
			for j in range (-1,2):
				num = matrix[centre[0]+i][centre[1]+j]
				temp = remove(num,temp)

		j = cell[0]
		l = cell[1]

		# 2. Row check
		row = cell[1]
		for i in range(9):
			temp = remove(matrix[i][row],temp)
			

		# 3. Column check
		column = cell[0]
		for i in range(9):
			temp = remove(matrix[column][i],temp)

		print "Tries: %s, TempLen: %s" % (tries, len(temp))

		cv2.rectangle(img2, (x+2+(w/9)*(l),y+2+(w/9)*(j)), (x-2+(w/9)*(l+1),y-2+(w/9)*(j+1)), [0,0,255], thickness=2, lineType=1, shift=0)
		if len(temp)==1:
			matrix[cell[0]][cell[1]]=temp[0]
			listofcells.remove(cell)
			cv2.putText(img2, str(temp[0]), (x+7+(w/9)*(l),y-7+(w/9)*(j+1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
			cv2.rectangle(img2, (x+2+(w/9)*(l),y+2+(w/9)*(j)), (x-2+(w/9)*(l+1),y-2+(w/9)*(j+1)), [255,0,0], thickness=2, lineType=1, shift=0)


		if len(listofcells)==0:
			completed = True

		#print (l,j,(x+2+(w/9)*(l),y+2+(w/9)*(j)), (x-2+(w/9)*(l+1),y-2+(w/9)*(j+1)))

		cv2.imshow('SUDOKU',img2)
		k = cv2.waitKey(1)
		if k & 0xFF==ord('q'):
			cv2.destroyAllWindows()
			break
		elif k & 0xFF==ord('n'):
			continue


		#		Now, check if tempNum array contains just one element, if so, then just put this element in the original matrix and remove its coordinates from listofcells matrix
		#					else: continue and check for another cell
		
		# 
	# When it comes out of the loop, hopefully the sudoku is solved
	print "Completed!"
	for i in range(9):
		print matrix[i]
	while(True):
		k = cv2.waitKey(0)
		if k & 0xFF==ord('q'):
			cv2.destroyAllWindows()
			break




def findBlock(cell, centres):
	minDist=100
	for i in range(9):
		dist = calcDist(cell,centres[i])
		if dist<minDist:
			minDist = dist
			index = i
	return index


def calcDist(x,y):
	dist = math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)
	return dist

def makeList(matrix):
	listofcells = []
	for i in range(9):
		for j in range(9):
			if(matrix[i][j]==0):
				listofcells.append([i,j])
	return listofcells

def remove(num, temp):
	if num in temp:
		temp.remove(num)
	return temp


def match(img):
	pred = 0
	#method = eval(meth)
	method = cv2.TM_CCOEFF_NORMED;
	for num in range(9):
		path=str('digits/'+str(num+1)+'.jpg')
		template = cv2.imread(path,0)
		#print path
		w, h = template.shape[::-1]

		# Apply template Matching
		res = cv2.matchTemplate(img,template,method)
		min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
		if(max_val>0.9):
			pred=num+1
			break
	return pred


if __name__ == '__main__':
	main()