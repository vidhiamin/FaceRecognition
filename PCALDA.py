# CS512 Computer Vision
# PROJECT
# Face Recognition using PCA and LDA.

from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial import distance
import pickle
import operator
import matplotlib.image as img
import math
from numpy import linalg as LA
import numpy as np

#######IMPLEMENTATION OF PCA#########
#Find the projection matrix for PCA and it also create the pickle file if it is not already exist.
def ProjectionMatrix(data_matrix, threshold, str):
    isthreshold = False
    number = 0
	#Find the data_matrix_centered
    datamatrixcentered = data_matrix - np.mean(data_matrix, axis=0)
	#Find the covariance
    data_matrix_cov = np.cov(datamatrixcentered, rowvar=False)
	#Find the eigenvalue and eigenVectors
    (eigenValues, eigenVectors) = LA.eigh(data_matrix_cov)
	#sorting the eigenvalues
    #Highest eigenvalues contain more information
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    #sum of eigenvalues
    total = np.sum(eigenValues)
    while isthreshold == False:
        sumofeigenValue = 0.0
        for i in range(eigenValues.size):
            sumofeigenValue = sumofeigenValue + eigenValues[i] / total
            number += 1
            if math.isclose(sumofeigenValue, threshold) or sumofeigenValue > threshold:
                isthreshold = True
                break
#It takes n number of eigenvectors from total eigenvectors
    projection_matrix = np.matrix([eigenVectors[n] for n in range(number)]).T                                
	#It checkes whether pickle file is not exist.If it is not exist then it create the file and write the projection matrix into it. 
    if not Path(str + '.pickle').exists():
        with open(str + '.pickle', 'wb') as handle:
            pickle.dump(projection_matrix, handle,protocol=pickle.HIGHEST_PROTOCOL)                      
    else:
		#If file is exist then it gives the error message that the file is already there.
        print ('Error.' + ' Another file with same name already found.')
    #return the projection matrix
    return projection_matrix

#for checking the accuracy for PCA.It will detect the eculiden distance between the train and test image.
def detectaccuracy(datamatrix_0,datamatrix_1,labelmatrix_0,labelmatrix_1):
    
    length = datamatrix_0.shape[0]
    map_ = np.zeros((1, length))
    labelmatrixnew = np.zeros((1, length))
    accuracymatrix = np.ones((1, length))

    for i in range(0, length):
        for j in range(0, length):
            
            #Find the eculiden distance.
            map_[0, j] = distance.euclidean(datamatrix_1[i], datamatrix_0[j])
        arg = map_.argmin()
        labelmatrixnew[0, i] = labelmatrix_0[arg]
        if labelmatrixnew[0, i] != labelmatrix_1[i]:
            accuracymatrix[0, i] = 0

    return 100 * np.sum(accuracymatrix) / np.size(accuracymatrix)

#defining PCA based on threshold(as it decides number of eigenvectors we need to consider to predict accuracy)

def PCA(data_matrix,proj_matrix,threshold,str):

    datamatrixcentered = data_matrix - np.mean(data_matrix, axis=0)
    if proj_matrix is None:
        projection_matrix = ProjectionMatrix(data_matrix, threshold, str)
    else:
        projection_matrix = proj_matrix  

    rd_data_matrix = np.matmul(datamatrixcentered, projection_matrix)
    return rd_data_matrix

#we initialised image matrix with zero.We took 10304 because dimensions of each image is 92x112(that is 10304).
imagematrix = np.zeros((0, 10304))

#Store the label of folders. from 1-40
temp = np.arange(1, 41, 1)

#It take the all label of all folders
label_matrix = np.array([[temp[i]] * 10 for i in range(temp.size)])

#Flatten that label so it will arrange in onw row only.
label_matrix = label_matrix.flatten()

#Give the path for folderwhere you put containing ORL dataset. We dont have to convert those image to grayscal because it is already grayscal image in databse.
folder = '../project/orl_faces/'
for j in range(1, 41):
    direction = folder + 's' + str(j) + '/'
    for i in range(1, 11):
		#it will take all images from all folders.
        directory = direction + str(i) + '.pgm'
		
		#Tranpose the image
        image = img.imread(directory).T
		
		#Flatten that image
        imagevector = np.asmatrix(image.flatten())
		
		#concatenate image with zero array.
        imagematrix = np.concatenate((imagematrix, imagevector))

#Split the train and test data
test_data_matrix = imagematrix[0:400:2]
training_data_matrix = imagematrix[1:400:2]

#split the train and test label.
label_test = label_matrix[0:400:2]
label_training = label_matrix[1:400:2]


##When you dont have any pickle file then just uncommented below code. It will create a pickle file and store the projection matrix.
#if pickle file is already there then just commented below line code.

#Pickle file for threshold value 0.95
#proj_data_mat=ProjectionMatrix(training_data_matrix,0.95,"proj_data_mat_0.95")

#Pickle file for threshold value 0.80
#proj_data_mat=ProjectionMatrix(training_data_matrix,0.80,"proj_data_mat_0.80")

#Pickle file for threshold value 0.90
#proj_data_mat=ProjectionMatrix(training_data_matrix,0.90,"proj_data_mat_0.90")

#Pickle file for threshold value 0.85
#proj_data_mat=ProjectionMatrix(training_data_matrix,0.85,"proj_data_mat_0.85")

#Here I took for values of threshold. You can take as many as you want. It will print accuracyof PCA for different threshold value.
threshold = np.matrix([[0.8,0.85,0.90,0.95]])

#it calculate the training data matrix and testing data matrix for threshold value = 0.8,0.85,0.90,0.95
for k in range(threshold.size):
    with open('proj_data_mat_' + str(threshold[0, k]) + '.pickle', 'rb') as handle:
        #Load the pickle files for threshold values 0.8, 0.85,0.90,0.95
        proj_data_mat = pickle.load(handle)
	
	#Calculate training data matrix for all threshold values
    training_data_matrix_rd = PCA(training_data_matrix, proj_data_mat,threshold[0, k], '')
    
	#Calculate test data matrix for all threshold values
    test_data_matrix_rd = PCA(test_data_matrix, proj_data_mat,threshold[0, k], '')
    
	#Print the accuracy.
    acc_prc = detectaccuracy(training_data_matrix_rd, test_data_matrix_rd,label_training, label_test)
                       
    print ('For threshold: ' + str(threshold[0, k]) + ' accuracy percentage= '  + str(acc_prc) + '%\n')
       
#########TESTING THE LABELS###############
#Now we will print the predicted label and actual label. and shows whether it is giving correctly or not.

#Take zeros matrix for (1,999)
map_ = np.zeros((1, 199))

#Now we will find the eculiden distance between the test data matrix and training data matrix.
for i in range(0,200):
    for j in range(0, 199):
        map_[0, j] = distance.euclidean(test_data_matrix_rd[i], training_data_matrix_rd[j])
    
    #It will find the index of minimum eculiden distance
    arg = map_.argmin()

    #print(arg)
    print("predicted label:",label_training[arg],"actual label :",label_training[i])

zero_index = img.imread("../project/orl_faces/s5/10.pgm").T

print("you shold put the image in following manner:(orl_faces/foldername(S(1-40))/imagename.pgm(1-10.pgm))")
inputIMage = input("Enter image path to predict: ")
im2 = img.imread(inputIMage).T

#Flatten the default image
test = np.asmatrix(zero_index.flatten())

#Flatten the test image
test1 = np.asmatrix(im2.flatten())


#proj =ProjectionMatrix(training_data_matrix,0.95,"xyz")

#concatenate with default image and test image
test = np.concatenate((test,test1),axis=0)

#Now we will find the accuracy of test image for threshold value 0.95. You can take any threshold value.
test = PCA(test, proj_data_mat,0.95, '')

###########END OF PCA IMPLEMENTATION##############


length = 1

#Take zero matrix with (1,200)
map_ = np.zeros((1, 200))

#Find the eculiden distance between one test image and all 200 test image and try to predict the label for test image
for j in range(0, 200):
    map_[0, j] = distance.euclidean(test[1], training_data_matrix_rd[j])

#It finds the index of the minimum eculiden distance 
arg = map_.argmin()

#Print the label for minimum eculiden distance

print('Image belong to folder number ',label_training[arg])
 
################LCA IMPLEMENTATTION################
#differenct people
numberOfclass=40

#dimentions of image. because the pixel size is 92 * 112 = 10304
imagedimention=10304

#Number of images in one folder
numberofimagesineachclass=10

#number of images in total
imagerows=400

#store the imagematrix into D
D=imagematrix

#We created an array to save the mean of every class(40,10304)
meanofclass=np.zeros((numberOfclass,imagedimention))

#finding individual class means
for i in range(numberOfclass):
    #adding evens because index of arrays start at zero not 1
    meanofclass[i]=np.mean(training_data_matrix[i*5:i*5+5],axis=0)
#print('individual class:', meanofclass.shape)

#Here nk represents number of training samples in a particular class
nk=numberofimagesineachclass // 2 

#overall sample mean
#get mean of each column and result is 1 row and 10304(dimentions) columns
meu=np.mean(D,axis=0)


#initializing in-between scatter matrix
Sb=np.zeros((imagedimention,imagedimention))
#calculating in between scattered matrix
for k in range(numberOfclass):
    difference_means=meanofclass[k]-meu
    difference_means=np.reshape(difference_means,(1,10304) )
    datam_tran=difference_means.transpose()
    B=np.matmul(datam_tran,difference_means)
    Sb+=nk*B
    
#Calculating mean 
Z=D
for i in range(numberOfclass):
    for j in range (numberofimagesineachclass):
        if(j % 2 == 0):
            Z[i*10+j]-=meanofclass[i][:]
print("shape of Z:",Z.shape) 

#Calculating with-in class scatter matrix 
Sw=np.zeros((imagedimention,imagedimention))
S_initial=np.zeros((numberofimagesineachclass // 2,imagedimention))
for i in range (numberOfclass):
    for j in range (numberofimagesineachclass // 1):
        if (j % 2== 0):    
            S_i=S_initial
            S_i[j // 2]= S_i[j // 2] +  Z[i*10+j]
    S_i=np.dot(S_i.T,S_i)
    Sw=Sw+S_i

#calculating inverse of with-in-class scatter matrix
Sw_inv= np.linalg.pinv(Sw)

#Obtaining Eigen vectors and values from sb and sw:
S_inv_mul_B=np.matmul(Sw_inv,Sb)
eigenvaluesl,eigenvectorsl = np.linalg.eig(S_inv_mul_B)
#sorting eigen values in descending order
sort_idx = np.argsort(eigenvaluesl)[::-1]
eigenvaluesl = eigenvaluesl[sort_idx]

eigenvectorsl = eigenvectorsl[:,sort_idx]


u=eigenvectorsl[:,:39]
lamp=eigenvaluesl[:39]


xtrain=training_data_matrix.dot(u)  
xtest=test_data_matrix.dot(u)


def accuracyL(predictedClasses, labelvector): # check predicted classes for correctness
    correct = 0
    for x in range(len(predictedClasses)):
        if predictedClasses[x] is labelvector[x]:
            correct += 1
    return (correct/float(len(predictedClasses))) * 100.0

#calculating accurcay for LDA considering nearest neighbour
def detectAccuracyLDA(trainingSet,labelvector,test,k):
    distances =[]
    for x in range(len(trainingSet)):
        #calculating distance between testing and training
        distance = np.linalg.norm(np.subtract(test,trainingSet[x]))
        #
        distances.append((labelvector[x],distance)) 
    distances.sort(key=operator.itemgetter(1)) #sorting distances array by the distance
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x]) 
    classes ={}
    for x in range(len(neighbors)): 
        response= neighbors[x][0] 
        if response in classes:
            classes[response] += 1 
        else:
            classes[response] = 1 
    sortedVotes = sorted(classes.items(), key=operator.itemgetter(1), reverse=True)#sort the class votes in decreasing order
    return sortedVotes[0][0] #return the most voted class number
count=1
Vectorlabel =[]
for x in range(40): #constructing the classes of each instance
    for j in range(5):
        Vectorlabel.append(count)
    count= count+1
k=[1]
accuracy=[]
for i in range(1):
    predictionslist=[] 
    for x in range(200):    #performing test on all the test_data_matrix
        predictionslist.append(detectAccuracyLDA(xtrain,Vectorlabel,xtest,k[i]))
    accuracy.append(accuracyL(predictionslist,Vectorlabel))
    print("Accuracy LDA percentage")
    print(accuracy[i],"%")



   

