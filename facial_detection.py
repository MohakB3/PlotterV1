from PIL import ImageFont, ImageDraw, Image
import cv2
import time
import numpy as np
import math
import subprocess

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

    #* Display the image + escape conditions
    cv2.imshow('Facial Recognition', cv2_im_processed)
    k = cv2.waitKey(30) & 0xff

    #* If a face is there for more than 5 seconds, start the image processing.
    if faceExists == True and startImageProcess == False:
        endTime = time.time()
        text = str(math.floor(endTime - startTime))
        if (endTime - startTime > 5):    
            print("Face Detected")
            text = "Image Processing Started"
            posX = 260

            if startImageProcess == False:
                cv2.imwrite("detectedImage.png", imgUnprocessed)

            startImageProcess = True
    elif faceExists == True and startImageProcess == True:
        text = "Image Processing Started"
    else:
        startTime = time.time()

    print("Does Face Exist:" + str(faceExists))

    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

#* Canny Edge Detection
image = cv2.imread('detectedImage.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5), 0)
canny = cv2.Canny(blur, 55, 100)

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

# TODO: Add a way to send the gcode read from the final_ouput file we saved to be sent over to the Ender 3 through serial.