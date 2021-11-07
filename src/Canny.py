import numpy as np
from scipy.ndimage.filters import convolve
import matplotlib.image as mpl
from PIL import Image, ImageTk
class Canny():
    def __init__(self) -> None:
        pass

    def Gaussian(self, size, sigma):
        trueSize = size//2
        xAxis, yAxis = np.mgrid[-trueSize:trueSize+1, -trueSize:trueSize+1]
        base = 1 / (2.0 * np.pi * sigma**2)
        filtered = np.exp(-((xAxis**2 + yAxis**2) / (2.0*sigma**2))) * base
        return filtered

    def sobel(self, image):
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
        
        Ix = convolve(image, Kx)
        Iy = convolve(image, Ky)
        
        magnitude = np.hypot(Ix, Iy)
        magnitude = magnitude / magnitude.max() * 255
        slope = np.arctan2(Iy, Ix)
        
        return (magnitude, slope)
    
    def GuasianAndSobel(self, BinaryImage):
        smoothImage = convolve(BinaryImage, self.Gaussian(5,1))
        gradientImage, angel = self.sobel(smoothImage)
        return gradientImage, angel

    def CannySupressionNonMax(self, image, direction):
        
        imageH = len(image)
        imageW = len(image[0])
        RImage = np.zeros((imageH,imageW), dtype=np.int32)
        angle = direction * 180. / np.pi
        angle[angle < 0] += 180

        
        for h in range(1,imageH-1):
            for w in range(1,imageW-1):
  
                q = 255
                r = 255
                
            #angle 0
                if (0 <= angle[h,w] < 22.5) or (157.5 <= angle[h,w] <= 180):
                    q = image[h, w+1]
                    r = image[h, w-1]
                #angle 45
                elif (22.5 <= angle[h,w] < 67.5):
                    q = image[h+1, w-1]
                    r = image[h-1, w+1]
                #angle 90
                elif (67.5 <= angle[h,w] < 112.5):
                    q = image[h+1, w]
                    r = image[h-1, w]
                #angle 135
                elif (112.5 <= angle[h,w] < 157.5):
                    q = image[h-1, w-1]
                    r = image[h+1, w+1]

                if (image[h,w] >= q) and (image[h,w] >= r):
                    RImage[h,w] = image[h,w]
                else:
                    RImage[h,w] = 0
        
        return RImage

    def GausSobAndSupression(self, BinaryImage):
        smoothImage = convolve(BinaryImage, self.Gaussian(5,1))
        gradientImage, angel = self.sobel(smoothImage)
        nonMaxImage = self.CannySupressionNonMax(gradientImage, angel)
        return nonMaxImage

    def CannyThreshold(self, image, lowThresholdRatio=0.05, highThresholdRatio=0.15):
    
        highThreshold = image.max() * highThresholdRatio
        lowThreshold = highThreshold * lowThresholdRatio
        
        imageH, imageW = image.shape
        returnImage = np.zeros((imageH,imageW), dtype=np.int32)
        
        weak = np.int32(75)
        strong = np.int32(255)
        
        strongX, stringY = np.where(image >= highThreshold)
        
        weakX, weakY = np.where((image <= highThreshold) & (image >= lowThreshold))
        
        returnImage[strongX, stringY] = strong
        returnImage[weakX, weakY] = weak
        
        return (returnImage, weak, strong)
    
    def GausSobNonMaxAndThreshold(self, BinaryImage):
        smoothImage = convolve(BinaryImage, self.Gaussian(5, 1))
        gradientImage, angel = self.sobel(smoothImage)
        nonMaxImage = self.CannySupressionNonMax(gradientImage, angel)
        threshImage = self.CannyThreshold(nonMaxImage)
        return threshImage

    def EdgeTacking(self, image, weak, strong=255):
        imageH, imageW = image.shape
        for h in range(1, imageH-1):
            for w in range (1, imageW-1):
                if(image[h,w] ==weak):
                    try:
                        if ((image[h+1, w-1] == strong) or (image[h+1, w] == strong) or (image[h+1, w+1] == strong)
                            or (image[h, w-1] == strong) or (image[h, w+1] == strong)
                            or (image[h-1, w-1] == strong) or (image[h-1, w] == strong) or (image[h-1, w+1] == strong)):
                            image[h, w] = strong
                    except IndexError as error:
                        pass
        return image

    def GausSobNonMaxThreshAndTrack(self, BinaryImage):
        smoothImage = convolve(BinaryImage, self.Gaussian(5, 1))
        gradientImage, angel = self.sobel(smoothImage)
        nonMaxImage = self.CannySupressionNonMax(gradientImage, angel)
        threshImage = self.CannyThreshold(nonMaxImage)
        cannyImage = self.EdgeTacking(threshImage[0],threshImage[1],threshImage[2])
        return cannyImage

    def CannyEdges(self,BinaryImage,imageH,imageW):

        smoothImage = convolve(BinaryImage, self.Gaussian(5, 1))
        gradientImage, angel = self.sobel(smoothImage)
        nonMaxImage = self.CannySupressionNonMax(gradientImage, angel)
        threshImage = self.CannyThreshold(nonMaxImage)
        cannyImage = self.EdgeTacking(threshImage[0],threshImage[1],threshImage[2])
        
        imageArray = np.empty([imageH, imageW], dtype=np.uint8)
        imageArray = np.asarray(cannyImage)
        displayImage = ImageTk.PhotoImage(Image.fromarray(imageArray))

        return displayImage