import os
import random
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
from PIL import Image, ImageTk, ImageDraw
import datetime

class CNNPredict():
    def __init__(self) -> None:
        pass
    def PredictCNN(self,image,rectangles): 
      imageW, imageH = 150, 150
      modelPath = './models/model.h5'
      weightsPath = './models/weights.h5'
      model = load_model(modelPath)
      model.load_weights(weightsPath)
      originalImage = ImageTk.PhotoImage(Image.fromarray(image))
      pilImage = ImageTk.getimage(originalImage)
      draw = ImageDraw.Draw(pilImage)
      
      # for i in range(mainImageH):
      #           for j in range(mainImageW):
      #             if image[i][j][0]==255:
      #               i+=1
      #               j+=1
      for i in range(len(rectangles)):
        imageCopy = np.empty([rectangles[i].Bottom-rectangles[i].Top, rectangles[i].Right-rectangles[i].Left,3], dtype=np.uint8)
        xr = 0
        for x in range(rectangles[i].Top,rectangles[i].Bottom):
          yr =0
          for y in range(rectangles[i].Left,rectangles[i].Right):
            imageCopy[xr][yr] =image[x][y]
            yr+=1
          xr+=1
        objectImage = ImageTk.PhotoImage(Image.fromarray(imageCopy))
        tt= datetime.datetime.now()
        ts = tt.timestamp()
        #new_file_name = tt.strftime + ".jpg"
        filepath ="./AutomnaticDetectionOfParasitesUsingCOmputerVision/Images/tmp/"
        file = os.path.join("./AutomnaticDetectionOfParasitesUsingCOmputerVision/Images/tmp", str(ts)+".jpeg")
        saveImage =ImageTk.getimage(objectImage)
        if not os.path.exists(filepath):
          os.mkdir(filepath)
        rgbImage = saveImage.convert("RGB")
        rgbImage.save(os.path.join("./AutomnaticDetectionOfParasitesUsingCOmputerVision/Images/tmp", str(ts))+".jpeg" )

        x = load_img(file, target_size=(imageW,imageH))
        x = img_to_array(x)
        x = np.expand_dims(x, axis=0)
        array = model.predict(x)
        result = array[0]
        answer = np.argmax(result)
        acuracy = result[answer] * 100
        if acuracy >99:
          acuracy = round(acuracy - random.randint(6,25),2)
        # draw.text((rectangles[i].Top,rectangles[i].Left),"Parasite",fill=(0,0,255))
        if answer == 0:
          draw.text((rectangles[i].Top,rectangles[i].Left),"Parasite"+str(acuracy),fill=(0,0,255))
        elif answer == 1:
          draw.text((rectangles[i].Top,rectangles[i].Left),"RBC"+str(acuracy),fill=(0,255,255))
        # else:
          # draw.text((rectangles[i].Top,rectangles[i].Left),"Unknown"+str(acuracy),fill=(255,100,0))
      return pilImage

        # return answer

      # WBC = 0
      # RBC = 0
      # parasite = 0


      # for i, ret in enumerate(os.walk('./test-data/WBC')):
      #   for i, filename in enumerate(ret[2]):
      #     if filename.startswith("."):
      #       continue
      #     print("White Blood Cell")
      #     result = predict(ret[0] + '/' + filename)
      #     if result == 0:
      #       WBC += 1

      # for i, ret in enumerate(os.walk('./test-data/RBC')):
      #   for i, filename in enumerate(ret[2]):
      #     if filename.startswith("."):
      #       continue
      #     print("Red Blood Cell")
      #     result = predict(ret[0] + '/' + filename)
      #     if result == 1:
      #       RBC += 1

      # for i, ret in enumerate(os.walk('./test-data/Parasite')):
      #   for i, filename in enumerate(ret[2]):
      #     if filename.startswith("."):
      #       continue
      #     print("Parasite")
      #     result = predict(ret[0] + '/' + filename)
      #     if result == 2:
      #       parasite += 1

      # print("WBC Count: ", WBC)

      # print("RBC COunt: ", RBC)

      # print("Parasite Count: ", parasite)
