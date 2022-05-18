from random import randrange 
import cv2

#Lets load some pretrained data on face frontals from OpenCV which used Haarcascade algorithm
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  #Here we created a classifier

#Choose an Image and read it 
#img = cv2.imread('Jeetu.jpeg')


#To capture vedio from Webcam
webcam = cv2.VideoCapture(0)

#iterate over vedio
while True:
    #Read the current frame
    success_frame_read, frame = webcam.read()
    gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #Convert into grayscale image
    face_coordinates = trained_face_data.detectMultiScale(gray_img)
    #Lets draw rectangle around all the faces there are in an image
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w , y+h), (randrange(256),randrange(256),randrange(256)), 5)
    cv2.imshow('face detector', frame)
    key = cv2.waitKey(1)   #Key variable stores the key which is pressed
    #Stop if Q key is pressed
    if key == 81 or key == 113:
        break

#Release the vedio capture object 
webcam.release()





#Now lets detect some faces
#face_coordinates = trained_face_data.detectMultiScale(gray_img)  

#Rem:- This face_coordinates will be a list which will have items inside it that are lists. that item list will have four values:- First two:- representing (x,y) value of one point(one corner) of the rectangle around the face and other two will be width and the height of the rectangle.
#So using that coordinate of one corner and by adding height and width to it's coordinates, we will get coordinates of another 3 corners also of the rectangle

#This face_coordinates is a list of lists. Each item in the main list will be list which will have 4values representing rectangle around each face.
#But if there is two faces in the image, then face_coordinates will be a list containing two lists, each representing rectangle over one face.







print("Code completed")