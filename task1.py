#PROJECT ( Image to pencil sketch with python )

#1st step - install cv2 libray ( code = pip install opencv2-python)

#2nd step ( import the library)
import cv2

#3rd step ( produce a variable )
image = cv2.imread("1.jpg")

#4th step ( convert into grey color)
grey_filter = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("grey-1.png", grey_filter)

#5th step ( invert the grey color image)
invert = cv2.bitwise_not(grey_filter)
cv2.imwrite("invert-1.png", invert)

#6th step(blur the image)
blur = cv2.GaussianBlur(invert, (21,21),0)
invertedblur = cv2.bitwise_not(blur)
cv2.imwrite("invertedblur-1.png" , invertedblur)

#Final step (pencil sketch)
pencilsketch = cv2.divide(grey_filter, invertedblur,scale=256.0)
cv2.imwrite("pencilsketch-1.png", pencilsketch) 

#completed





