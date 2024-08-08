from PIL import ImageFont, ImageDraw, Image
import cv2
import time
import numpy as np
import math
import subprocess
import tkinter as tk
from tkinter import ttk

#* Basic Facial Recognition
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1290)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

faceExists = False
startImageProcess = False

startTime = time.time()

text = "Loading"
posX = 100
textColor = (102, 158, 239)

#* Create Tkinter Window for UI Controls
window = tk.Tk()
window.title("Edge Detection Values")

label1 = tk.Label(window, text = 'Canny Threshold In')
label1.pack()

cannyRampIn = tk.IntVar(value = 55)
rampInScale = ttk.Scale(
    window, 
    command = lambda value: print(cannyRampIn.get()), 
    from_ = 0, 
    to = 200,
    length = 300, 
    variable = cannyRampIn)
rampInScale.pack()

label2 = tk.Label(window, text = 'Canny Threshold Out')
label2.pack()

cannyRampOut = tk.IntVar(value = 100)
rampOutScale = ttk.Scale(
    window, 
    command = lambda value: print(cannyRampOut.get()), 
    from_ = 0, 
    to = 200,
    length = 300, 
    variable = cannyRampOut)
rampOutScale.pack()

#* Image Stacking Algorithm
def stackImages(scale,imgArray):

    rows = len(imgArray)

    cols = len(imgArray[0])

    rowsAvailable = isinstance(imgArray[0], list)

    width = imgArray[0][0].shape[1]

    height = imgArray[0][0].shape[0]

    if rowsAvailable:

        for x in range ( 0, rows):

            for y in range(0, cols):

                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:

                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)

                else:

                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale,scale)

                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)

        imageBlank = np.zeros((height, width, 3), np.uint8)

        hor = [imageBlank]*rows

        hor_con = [imageBlank]*rows

        for x in range(0, rows):

            hor[x] = np.hstack(imgArray[x])

        ver = np.vstack(hor)

    else:

        for x in range(0, rows):

            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:

                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)

            else:

                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)

            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)

        hor = np.hstack(imgArray)

        ver = hor

    return ver

while True:
    #* Reading each frame and detecting if there's a face
    _, img = cap.read()
    _, imgUnprocessed = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    #* Drawing a rectangle where a face is detected
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (160, 207, 200), 2)
        cv2.circle(img, (x+w, y+h), int(w/30), (168, 171, 120), int(w/20))
        cv2.circle(img, (x+w, y), int(w/30), (168, 171, 120), int(w/20))
        cv2.circle(img, (x, y), int(w/30), (168, 171, 120), int(w/20))
        cv2.circle(img, (x, y+h), int(w/30), (168, 171, 120), int(w/20))

    #* Check for whethere there is a face or not
    if len(faces) > 0:
        faceExists = True
    else:
        faceExists = False

    #* img = cv2.putText(img, text, textCoords, textFont, textFontScale, textColor, textThickness, cv2.LINE_AA)
    pil_im = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_im)
    font = ImageFont.truetype("KGDefyingGravity.ttf", 80)
    draw.text((posX, 100), text, font=font, fill=textColor)
    cv2_im_processed = np.array(pil_im)

    #* Visual Canny Edge Detection
    grayInternal = cv2.cvtColor(imgUnprocessed, cv2.COLOR_BGR2GRAY)
    blurInternal = cv2.GaussianBlur(grayInternal,(5,5), 0)
    cannyInternal = cv2.Canny(blurInternal, cannyRampIn.get(), cannyRampOut.get())

    #* Combine image output + display the image + escape conditions
    StackedImages = stackImages(0.5,([cv2_im_processed,cannyInternal]))
    cv2.imshow('Facial Recognition', StackedImages)
    k = cv2.waitKey(30) & 0xff

    #* If a face is there for more than 5 seconds, start the image processing.
    if faceExists == True and startImageProcess == False:
        endTime = time.time()
        text = str(math.floor(endTime - startTime))
        if (endTime - startTime > 5):    
            print("Face Detected")
            text = "Release to Start Image Processing"
            posX = 130

            if startImageProcess == False:
                cv2.imwrite("detectedImage.png", imgUnprocessed)

            startImageProcess = True
    elif faceExists == True and startImageProcess == True:
        text = "Release to Start Image Processing"
    else:
        startTime = time.time()

    print("Does Face Exist:" + str(faceExists))

    #* Tkinter Mainloop Replacement
    window.update_idletasks()
    window.update()

    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

#* Canny Edge Detection
image = cv2.imread('detectedImage.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5), 0)
canny = cv2.Canny(blur, cannyRampIn.get(), cannyRampOut.get())

cv2.imwrite("Canny.png", canny)

#* Graphics Magick
subprocess.call('magick Canny.png Canny.pnm', shell=True)

#* Potrace
subprocess.call('potrace Canny.pnm -s -o Canny.svg', shell=True)

#* svg2gcode
subprocess.call('cargo run --release -- --dpi "300" --origin "0,200" --feedrate "1500" Canny.svg -o out.gcode', shell=True)

#* Modify Gcode to add Z-Hops
lookup = 'G0'
mostRecentGZero = 0

with open('out.gcode', 'r') as inString:
    with open('output.gcode', 'w') as outString:
        for num, line in enumerate(inString, 1):
            if lookup in line:
                line = line.rstrip('\n')
                line += ' Z3'
                mostRecentGZero = num
                print(line, file=outString)
            elif num == (mostRecentGZero + 1):
                line = line.rstrip('\n')
                line += ' Z0'
                print(line, file=outString)
            else:
                line = line.rstrip('\n')
                print(line, file=outString)

with open('final_output.gcode', 'w') as outfile:
    with open('output.gcode', 'r') as infile1:
        with open('logo_output.gcode', 'r') as infile2:
            outfile.write(infile1.read() + '\n')
            outfile.write(infile2.read() + '\n')

#* Send the G-Code to the Ender3 through pyGcodeSender
subprocess.call('python pyGcodeSender.py final_output.gcode', shell=True)