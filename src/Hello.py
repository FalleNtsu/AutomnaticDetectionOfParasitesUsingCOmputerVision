# img_viewer.py

from typing import IO
from PIL import Image, ImageTk
import PySimpleGUI as sg
import os.path
import numpy as np
import ImageProcessing as imageProccess
import matplotlib.image as mpl
import io

# First the window layout in 2 columns
imp = imageProccess.ImageProcessing()
file_list_column = [
    [
        sg.Text("Image Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
        )
    ],
    [
        sg.Button(button_text = "Convert To GreyScale", key = "-GREYSCALE-")
    ],
    [
        sg.Button(button_text = "Convert To Binary", key = "-BINARYSCALE-")
    ],
    [
        sg.Button(button_text = "Find Objects", key = "-FINDOBJECTSSLIDER-")
    ],
    [
        sg.Button(button_text = "Canny", key = "-CANNY-")
    ],
     [
        sg.Button(button_text = "HOG", key = "-HOG-")
    ],
     [
        sg.Button(button_text = "Find Objects Using CV2", key = "-FindObjectsLibraries-")
    ],
      [
        sg.Button(button_text = "Train CNN", key = "-TrainCNN-")
    ],
      [
        sg.Button(button_text = "Predict CNN", key = "-PredictCNN-")
    ],
]

# For now will only show the name of the file that was chosen
image_viewer_column = [
    [sg.Text("Choose an image from list on left:")],
    [sg.Text(size=(40, 1), key="-TOUT-")],
    [sg.Image(key="-IMAGE-")],
]
converted_image_viewer_column = [
    [sg.Text("The convertedImage shows here:")],
    [sg.Text(size=(40, 1), key="-CONVERTEDTOUT-")],
    [sg.Image(key="-CONVERTEDIMAGE-")],
]

# ----- Full layout -----
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
        sg.VSeperator(),
        sg.Column(converted_image_viewer_column),
    ]
]

window = sg.Window("Image Viewer", layout)

# Run the Event Loop
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    # Folder name was filled in, make a list of files in the folder
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".png", ".jpg"))
        ]
        window["-FILE LIST-"].update(fnames)
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        try:
            filename = os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0]
            )
            if filename.lower().endswith(".jpg"):

                bio = imp.ConvertJPGtoPNG(filename)
                window["-IMAGE-"].update(data=bio.getvalue())
                window["-TOUT-"].update(filename)
            else:
                window["-TOUT-"].update(filename)
                window["-IMAGE-"].update(filename=filename)

        except:

            pass
    elif event == "-GREYSCALE-":
        try:
            filename = os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0]
            )
            displayImage = imp.ConvertImageToGreyScale(filename)

            window["-CONVERTEDIMAGE-"].update(data=displayImage)
        except:
            pass
    elif event == "-BINARYSCALE-":
        try:
            filename = os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0]
            )
            displayImage = imp.ConvertToBinaryImage(filename)

            window["-CONVERTEDIMAGE-"].update(data=displayImage)
        except:
            pass
    elif event == "-FINDOBJECTSSLIDER-":
        # try:
            filename = os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0]
            )
            displayImage = imp.ExtractFeatures(filename)

            window["-CONVERTEDIMAGE-"].update(data=displayImage)
        # except:
        #     pass
    elif event == "-CANNY-":
        # try:
            filename = os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0]
            )
            displayImage = imp.CannyEdges(filename)

            window["-CONVERTEDIMAGE-"].update(data=displayImage)
        # except:
        #     pass
    elif event == "-FindObjectsLibraries-":
        # try:
            filename = os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0]
            )
            displayImage = imp.FindBoxesUsingOpenCV(filename)

            window["-CONVERTEDIMAGE-"].update(data=displayImage)
        # except:
        #     pass
    elif event == "-HOG-":
        # try:
            filename = os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0]
            )
            displayImage = imp.HOG(filename)

            window["-CONVERTEDIMAGE-"].update(data=displayImage)
        # except:
        #     pass
    elif event == "-TrainCNN-":
        # try:
            filename = os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0]
            )
            # displayImage = imp.pwd(filename)
            imp.trainCNN()

            # window["-CONVERTEDIMAGE-"].update(data=displayImage)
        # except:
        #     pass
    elif event == "-PredictCNN-":
        # try:
            filename = os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0]
            )
            displayImage = imp.predictCNN(filename)

            window["-CONVERTEDIMAGE-"].update(data=displayImage)
        # except:
        #     pass

window.close()

