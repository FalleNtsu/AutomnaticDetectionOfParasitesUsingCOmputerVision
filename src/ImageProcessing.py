import PIL
import numpy as np
from PIL import Image, ImageTk
import matplotlib.image as mpl
import io
import cv2 as cv
from SlidingWindowObject import SlidingWindowObject
import scipy
from scipy.ndimage.filters import convolve

class ImageProcessing():
    def __init__(self) -> None:
        pass

    def ConvertImageToGreyScale(self, filename):
        image =mpl.imread(filename)
        imageH = len(image)
        imageW = len(image[0])
        grayImage = np.empty([imageH, imageW], dtype=np.uint8)
        for i in range(imageH):
            for j in range(imageW):
                grayImage[i][j] = int(image[i][j][0]*0.2126 + image[i][j][1]*0.7152 + image[i][j][2] * 0.0722)
        displayImage = ImageTk.PhotoImage(Image.fromarray(grayImage))

        return displayImage

    def ConvertImageToGreyScalePNG(self, filename):
        image =mpl.imread(filename)
        imageH = len(image)
        imageW = len(image[0])
        grayImage = np.empty([imageH, imageW], dtype=np.uint8)
        for i in range(imageH):
            for j in range(imageW):
                grayImage[i][j] = int(image[i][j][0]*0.2126 + image[i][j][1]*0.7152 + image[i][j][2] * 0.0722)
        displayImage = ImageTk.PhotoImage(Image.fromarray(grayImage))

        return displayImage
    
    def ConvertJPGtoPNG(self, filename):
        image = Image.open(filename)
        image.thumbnail((400, 400))
        bio = io.BytesIO()
        image.save(bio, format="PNG")

        return bio
    
    def ConvertToBinaryImage(self, filename):
        image =mpl.imread(filename)
        imageH = len(image)
        imageW = len(image[0])
        BinaryImage = np.empty([imageH, imageW], dtype=np.uint8)
        grayImage = np.empty([imageH, imageW], dtype=np.uint8)
        for i in range(imageH):
            for j in range(imageW):
                newint = int(image[i][j][0]*0.2126 + image[i][j][1]*0.7152 + image[i][j][2] * 0.0722)
                grayImage[i][j] = newint
                if grayImage[i][j]>195:
                    BinaryImage[i][j] = 0
                else:
                    BinaryImage[i][j] = 255

        displayImage = ImageTk.PhotoImage(Image.fromarray(BinaryImage))

        return displayImage
    
    def ExtractFeatures(self, filename):
        image =mpl.imread(filename)
        imageH = len(image)
        imageW = len(image[0])
        BinaryImage = np.empty([imageH, imageW], dtype=np.uint8)
        grayImage = np.empty([imageH, imageW], dtype=np.uint8)
        for i in range(imageH):
            for j in range(imageW):
                newint = int(image[i][j][0]*0.2126 + image[i][j][1]*0.7152 + image[i][j][2] * 0.0722)
                grayImage[i][j] = newint
                if grayImage[i][j]>123:
                    BinaryImage[i][j] = 0
                else:
                    BinaryImage[i][j] = 255
# create the window
        window = np.empty([8, 8])
        objectArray = []
        for i in range(0, imageH, 8):
            nexti = i+8
            for j in range(0, imageW, 8):
                if self.checkIfvaluesInObjectArray(objectArray,i,j):
                    nextj = j+8
                    if nextj < imageW and nexti < imageH:
                        window = self.AddToWindow(BinaryImage,i,j,8)
                        windowValue = self.WindowSum(window,8)
                        
                        #check if there is a white value in the window
                        if windowValue>0:
                            
                            height = 0
                            width = 0
                            top = 0
                            bottom = 0
                            left = 0
                            LeftY =0
                            right = 0
                            rightY=0
                            #calculate the height of the object
                            top = self.GetopCellOfObjectFromWindow(window,i,j, 8)
                            for y in range(top, imageH):
                                if BinaryImage[y][j] == 255:
                                    for x in range(j,imageW):
                                        if BinaryImage[y][x] == 255:
                                            if right <= x:
                                                right = x
                                        if BinaryImage[y][x] == 0:
                                            if j ==0:
                                                left =0
                                            else:
                                                left = j - (x - j)
                                            break
                                else:
                                    bottom = (y-i)*2
                                    break
                            

                            
                            # for y in range(i, imageH, 4):
                            #     nexty = y+8
                            #     if nexty < imageH and nextj < imageW:
                            #         smallerWindowY = self.AddToWindow(BinaryImage,y,j,4)
                            #         smallerwindowValueY = self.WindowSum(smallerWindowY,4)
                            #         nextWindow = self.AddToWindow(BinaryImage,y+4,j,4)
                            #         nextValue = self.WindowSum(nextWindow, 4)

                            #     if smallerwindowValueY>0 and smallerwindowValueY<=nextValue:
                            #         height+=4
                            #     else:
                            #         break
                                    
                            # for x in range(j, imageW, 4):
                            #     nextx = x+8
                            #     #get the half way point of the height
                            #     midPointY = int(height/2)
                                
                            #     if nextx < imageW and nexti < imageH :
                            #         smallerWindowX = self.AddToWindow(BinaryImage,i+midPointY,x,4)
                            #         smallerwindowValueX = self.WindowSum(smallerWindowX,4)
                            #         nextWindow = self.AddToWindow(BinaryImage,i+midPointY,x,4)
                            #         nextValue = self.WindowSum(nextWindow, 4)

                            #     if smallerwindowValueX>0 and smallerwindowValueX>=nextValue:
                            #         width+=4
                            #     else:
                            #         break
                            # topleft = [i, j-width]
                            # topright = [i, j+width]
                            # bottomleft = [i-height, j-width]
                            # bottomright = [i-height, j+width]
                            if left != 0 and right != 0:
                                swo = SlidingWindowObject(top,bottom,left,right)
                                objectArray.append(swo)
                                imageCopy = np.array(image)
        ImageWithBoxes = self.DrawBoxes(objectArray,imageCopy)


                        
                    
        displayImage = ImageTk.PhotoImage(Image.fromarray(ImageWithBoxes))

        return displayImage
    
    def AddToWindow(self, array, currentX, currentY, windowSize):
        window = np.empty([windowSize, windowSize])
        c=0
        for y in range(currentY,currentY+windowSize,1):
            r=0
            
            for x in range(currentX,currentX+windowSize,1):

                if array[currentX+r][currentY+c] == 255:
                    window[c][r] = 1
                    r+=1
                else:
                    window[c][r] = 0
                    r+=1
            c+=1
        return window

    def WindowSum(self, window, windowSize):
        windowTotal = 0
        for x in range(windowSize):
            for y in range(windowSize):
                windowTotal += window[x][y]
        
        return windowTotal
    
    def checkIfvaluesInObjectArray(self,objectArray, i, j):
        if len(objectArray)>0:
            for x in objectArray:
                if x.Top <= i <= x.Bottom and x.Left <= j <= x.Right:
                    return False
                else:
                    return True
        return True

    def GetopCellOfObjectFromWindow(self, window, i,j, windowSize):
        top = -1
        if len(window)>0:
            for x in range(windowSize):
                for y in range(windowSize):
                    if window[x][y] == 1:
                        top = x+i
                        break
                if top >= i:
                    break
        return top
    
    def DrawBoxes(self, objectArray, OriginalImage):
        imageH = len(OriginalImage)
        imageW = len(OriginalImage[0])
        for x in objectArray:
            #draw top line
            for c in range(x.Left,x.Right):
                OriginalImage[x.Top][c][0] = 255
                OriginalImage[x.Top][c][1] = 0
                OriginalImage[x.Top][c][2] = 0
                #draw Bottom line       
                OriginalImage[x.Bottom][c][0] = 255
                OriginalImage[x.Bottom][c][1] = 0
                OriginalImage[x.Bottom][c][2] = 0
                #draw Left Line
            for c in range(x.Top,x.Bottom):
                OriginalImage[c][x.Left][0] = 255
                OriginalImage[c][x.Left][1] = 0
                OriginalImage[c][x.Left][2] = 0
                #draw Right line       
                OriginalImage[c][x.Right][0] = 255
                OriginalImage[c][x.Right][1] = 0
                OriginalImage[c][x.Right][2] = 0
        return OriginalImage

    def Gaussian(self, size, sigma):
        trueSize = size//2
        xAxis, yAxis = np.mgrid[-trueSize:trueSize+1, -trueSize:trueSize+1]
        base = 1 / (2.0 * np.pi * sigma**2)
        filtered = np.exp(-((xAxis**2 + yAxis**2) / (2.0*sigma**2))) * base
        return filtered

    def sobel(self, img):
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
        
        Ix = convolve(img, Kx)
        Iy = convolve(img, Ky)
        
        magnitude = np.hypot(Ix, Iy)
        magnitude = magnitude / magnitude.max() * 255
        slope = np.arctan2(Iy, Ix)
        
        return (magnitude, slope)

    def CannySupressionNonMax(self, image, direction):
        
        imageH = len(image)
        imageW = len(image[0])
        RImage = np.zeros((imageH,imageW), dtype=np.int32)
        angle = direction * 180. / np.pi
        angle[angle < 0] += 180

        
        for i in range(1,imageH-1):
            for j in range(1,imageW-1):
                try:
                    q = 255
                    r = 255
                    
                #angle 0
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        q = image[i, j+1]
                        r = image[i, j-1]
                    #angle 45
                    elif (22.5 <= angle[i,j] < 67.5):
                        q = image[i+1, j-1]
                        r = image[i-1, j+1]
                    #angle 90
                    elif (67.5 <= angle[i,j] < 112.5):
                        q = image[i+1, j]
                        r = image[i-1, j]
                    #angle 135
                    elif (112.5 <= angle[i,j] < 157.5):
                        q = image[i-1, j-1]
                        r = image[i+1, j+1]

                    if (image[i,j] >= q) and (image[i,j] >= r):
                        RImage[i,j] = image[i,j]
                    else:
                        RImage[i,j] = 0

                except IndexError as e:
                    pass
        
        return RImage

    def CannyThreshold(self, image, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    
        highThreshold = image.max() * highThresholdRatio
        lowThreshold = highThreshold * lowThresholdRatio
        
        imageH, imageW = image.shape
        returnImage = np.zeros((imageH,imageW), dtype=np.int32)
        
        weak = np.int32(25)
        strong = np.int32(255)
        
        strongX, stringY = np.where(image >= highThreshold)
        
        weakX, weakY = np.where((image <= highThreshold) & (image >= lowThreshold))
        
        returnImage[strongX, stringY] = strong
        returnImage[weakX, weakY] = weak
        
        return (returnImage, weak, strong)

    def CannyEdges(self,filename):
        image =mpl.imread(filename)
        imageH = len(image)
        imageW = len(image[0])
        grayImage = np.empty([imageH, imageW], dtype=np.uint8)
        for i in range(imageH):
            for j in range(imageW):
                grayImage[i][j] = int(image[i][j][0]*0.2126 + image[i][j][1]*0.7152 + image[i][j][2] * 0.0722)

        

        smoothImage = convolve(grayImage, self.Gaussian(5, 1))
        gradientImage, angel = self.sobel(smoothImage)
        nonMaxImage = self.CannySupressionNonMax(gradientImage, angel)
        threshImage = self.CannyThreshold(nonMaxImage)
        imageArray = np.empty([imageH, imageW], dtype=np.uint8)
        imageArray = np.asarray(threshImage[0])
        displayImage = ImageTk.PhotoImage(Image.fromarray(imageArray))

        return displayImage
    
    def FindBoxesUsingOpenCV(self, filename):
            image =mpl.imread(filename)
            imageCopy = np.array(image)
            imageH = len(image)
            imageW = len(image[0])
            # BinaryImage = np.empty([imageH, imageW], dtype=np.uint8)
            # grayImage = np.empty([imageH, imageW], dtype=np.uint8)
            grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            (threshold, BinaryImage) = cv.threshold(grayImage, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
            # for i in range(imageH):
            #     for j in range(imageW):
            #         newint = int(image[i][j][0]*0.2126 + image[i][j][1]*0.7152 + image[i][j][2] * 0.0722)
            #         grayImage[i][j] = newint
            #         if grayImage[i][j]>195:
            #             BinaryImage[i][j] = 0
            #         else:
            #             BinaryImage[i][j] = 255
            
            
                # threshold = 195
    
            cvCanny = cv.Canny(BinaryImage, threshold, threshold * 2)   
            
            edges, _ = cv.findContours(cvCanny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            
            
            objectEdges = [None]*len(edges)
            rect = [None]*len(edges)

            for i, c in enumerate(edges):
                objectEdges[i] = cv.approxPolyDP(c, 3, True)
                rect[i] = cv.boundingRect(objectEdges[i])

            drawing = np.zeros((cvCanny.shape[0], cvCanny.shape[1], 3), dtype=np.uint8)
            
            for i in range(len(edges)):
                color = (255, 0, 0)
                cv.rectangle(drawing, (int(rect[i][0]), int(rect[i][1])), (int(rect[i][0]+rect[i][2]), int(rect[i][1]+rect[i][3])), color, 1)
            
            for i in range(imageH):
                for j in range(imageW):
                    if drawing[i][j][0]==255:
                        imageCopy[i][j]=drawing[i][j]
                    

            displayImage = ImageTk.PhotoImage(Image.fromarray(imageCopy))
            return displayImage