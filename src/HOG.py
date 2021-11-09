from skimage.feature import hog
import numpy as np
import cv2
from collections.abc import Iterable
class SliddingWindow():
    def __init__(self) -> None:
        pass

    def HOGgetFeatures(self,img, orientation, pxlPCell, cellPerWindow,
                        vis=False, featureVectors=True):
   
        if vis == True:
            features, hog_image = hog(img, orientations=orientation,
                                    pixels_per_cell=(pxlPCell, pxlPCell),
                                    cells_per_block=(cellPerWindow, cellPerWindow),
                                    transform_sqrt=False,
                                    visualise=vis, feature_vector=featureVectors)
            return features, hog_image
        
        else:      
            features = hog(img, orientations=orientation,
                        pixels_per_cell=(pxlPCell, pxlPCell),
                        cells_per_block=(cellPerWindow, cellPerWindow),
                        transform_sqrt=False,
                        visualise=vis, feature_vector=featureVectors)
            return features
    
    def HOGgradient(channel):
        gradRow = np.empty(channel.shape, dtype=channel.dtype)
        gradRow[0, :] = 0
        gradRow[-1, :] = 0
        gradRow[1:-1, :] = channel[2:, :] - channel[:-2, :]
        gradCol = np.empty(channel.shape, dtype=channel.dtype)
        gradCol[:, 0] = 0
        gradCol[:, -1] = 0
        gradCol[:, 1:-1] = channel[:, 2:] - channel[:, :-2]
 
        return gradRow, gradCol

    def hog(this, image, orientations=9, cellPix=(8, 8), blockCells=(3, 3),
        block_norm='L2-Hys', visualize=False, transform_sqrt=False,
        feature_vector=True, multichannel=None, *, channel_axis=None):

        image = np.atleast_2d(image)
        float_dtype = this.checkFloatType(image.dtype)
        image = image.astype(float_dtype, copy=False)

        multichannel = channel_axis is not None  

        if transform_sqrt:
            image = np.sqrt(image)

        if multichannel:
            g_row_by_ch = np.empty_like(image, dtype=float_dtype)
            g_col_by_ch = np.empty_like(image, dtype=float_dtype)
            g_magn = np.empty_like(image, dtype=float_dtype)

            for idx_ch in range(image.shape[2]):
                g_row_by_ch[:, :, idx_ch], g_col_by_ch[:, :, idx_ch] = \
                    this.HOGgradient(image[:, :, idx_ch])
                g_magn[:, :, idx_ch] = np.hypot(g_row_by_ch[:, :, idx_ch],
                                                g_col_by_ch[:, :, idx_ch])

            # get pix gradient
            idcs_max = g_magn.argmax(axis=2)
            rr, cc = np.meshgrid(np.arange(image.shape[0]),
                                np.arange(image.shape[1]),
                                indexing='ij',
                                sparse=True)
            gradRow = g_row_by_ch[rr, cc, idcs_max]
            gradCol = g_col_by_ch[rr, cc, idcs_max]
        else:
            gradRow, gradCol = this.HOGgradient(image)
            startRow, startCol = image.shape[:2]
            cellRow, cellCol = cellPix
            blockRow, clockCol = blockCells

            numberofRows = int(startRow // cellRow)  
            numberofCols = int(startCol // cellCol)  

            # get orienatations
            orientation_histogram = np.zeros((numberofRows, numberofCols, orientations),
                                            dtype=float)
            gradRow = gradRow.astype(float, copy=False)
            gradCol = gradCol.astype(float, copy=False)

            this.getHistograms(gradCol, gradRow, cellCol, cellRow, startCol, startRow,
                                        numberofCols, numberofRows,
                                        orientations, orientation_histogram)
            if visualize:
                from .. import draw

                radius = min(cellRow, cellCol) // 2 - 1
                OrientationsArray = np.arange(orientations)
               
                orientationMidPoints = (
                    np.pi * (OrientationsArray + .5) / orientations)
                dr_arr = radius * np.sin(orientationMidPoints)
                dc_arr = radius * np.cos(orientationMidPoints)
                HOGimage = np.zeros((startRow, startCol), dtype=float_dtype)
                for r in range(numberofRows):
                    for c in range(numberofCols):
                        for o, dr, dc in zip(OrientationsArray, dr_arr, dc_arr):
                            centre = tuple([r * cellRow + cellRow // 2,
                                            c * cellCol + cellCol // 2])
                            rr, cc = draw.line(int(centre[0] - dc),
                                            int(centre[1] + dr),
                                            int(centre[0] + dc),
                                            int(centre[1] - dr))
                            HOGimage[rr, cc] += orientation_histogram[r, c, o]

            blocksRow = (numberofRows - blockRow) + 1
            blocksCol = (numberofCols - clockCol) + 1
            nomalizedBlocks = np.zeros(
                (blocksRow, blocksCol, blockRow, clockCol, orientations),
                dtype=float_dtype
            )

            for r in range(blocksRow):
                for c in range(blocksCol):
                    block = orientation_histogram[r:r + blockRow, c:c + clockCol, :]
                    nomalizedBlocks[r, c, :] = this.Normalize(block, method=block_norm)

            if feature_vector:
                nomalizedBlocks = nomalizedBlocks.ravel()

            if visualize:
                return nomalizedBlocks, HOGimage
            else:
                return nomalizedBlocks
    
    def Normalize(block, method, eps=1e-5):
        if method == 'L1':
            out = block / (np.sum(np.abs(block)) + eps)
        elif method == 'L1-sqrt':
            out = np.sqrt(block / (np.sum(np.abs(block)) + eps))
        elif method == 'L2':
            out = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
        elif method == 'L2-Hys':
            out = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
            out = np.minimum(out, 0.2)
            out = out / np.sqrt(np.sum(out ** 2) + eps ** 2)
     
            

        return out

    def getHistograms(this, gradColumns,
        gradient_rows,
        columns,  rows,
        columnSize,  rowSize,
        numColumCells,  NumCellsPerRow,
        NumOrientations,
        histOri):

        magnitude = np.hypot(gradColumns, gradient_rows)
        orientation = np.rad2deg(np.arctan2(gradient_rows, gradColumns)) % 180
        

        rowS = rows / 2
        colS = columns / 2
        cc = rows * NumCellsPerRow
        cr = columns * numColumCells
        ranRowsE = (rows + 1) / 2
        rowsRangeS = -(rows / 2)
        columnRangeE = (columns + 1) / 2
        columnsRangeS = -(columns / 2)
        orientations180 = 180. / NumOrientations

        
        
        for i in range(NumOrientations):
           
            oristart = orientations180 * (i + 1)
            oriend = orientations180 * i
            c = colS
            r = rowS
            rowIndex = 0
            colIndex = 0

            while r < cc:
                colIndex = 0
                c = colS

                while c < cr:
                    histOri[rowIndex, colIndex, i] = \
                        this.cell_hog(magnitude, orientation,
                                oristart, oriend,
                                columns, rows, c, r,
                                columnSize, rowSize,
                                rowsRangeS, ranRowsE,
                                columnsRangeS, columnRangeE)
                    colIndex += 1
                    c += columns

                rowIndex += 1
                r += rows

                
    def cell_hog( magnitude, orientation, oriStart,  
                            oriEnd, ColCells,  rowsCells,
                            colIndex,  RowIndex,
                            sizeCols,  sizeRows,
                            RangeRowsS, rangRowsE,
                            randCOlsS,  RangeColsE):

        total = 0

        for row in range(RangeRowsS, rangRowsE):
            Rindex = RowIndex + row
            if (Rindex < 0 or Rindex >= sizeRows):
                continue

            for col in range(randCOlsS, RangeColsE):
                colIndex = colIndex + col
                if (colIndex < 0 or colIndex >= sizeCols
                        or orientation[Rindex, colIndex] >= oriStart
                        or orientation[Rindex, colIndex] < oriEnd):
                    continue

                total += magnitude[Rindex, colIndex]

            return total / (rowsCells * ColCells)
    
    def checkFloatType(this,inputType, allowCom=False):
        if isinstance(inputType, Iterable) and not isinstance(inputType, str):
            return np.result_type(*(this.checkFloatType(d) for d in inputType))

        inputType = np.dtype(inputType)
        if not allowCom and inputType.kind == 'c':
            raise ValueError("wrong float type")

        return this.new_float_type.get(inputType.char, np.float64)
    
    new_float_type = {

    np.float32().dtype.char: np.float32,
    np.float64().dtype.char: np.float64,
    np.complex64().dtype.char: np.complex64,
    np.complex128().dtype.char: np.complex128,

    np.float16().dtype.char: np.float32,
    'g': np.float64,      
    'G': np.complex128,   
    }


    def HOGBoundingboxes(self, filename):
        image =mpl.imread(filename)
        imageCopy = np.array(image)
        imageH = len(image)
        imageW = len(image[0])
        hogFeatures ,displayImasge = hog(image, visualize=True)
        right, left, top, bottom = 0
        objectArray = []
        for r in range (0,imageW,9):
            for c in range (0,imageW,9):
                if r not in objectArray and c not in objectArray:
                    if hogFeatures.array[r][c][0] > 0.55:
                        top, left, bottom, right = self.hogFindObject(self, hogFeatures.array, r, c, imageH, imageW)
                        box = SlidingWindowObject(top,bottom,left,right)
                        objectArray.append(box)
        for i in range(len(objectArray)):
            color = (255, 0, 0)
            
            #left top right bottom
            cv.rectangle(displayImasge, (int(objectArray.left), int(objectArray.top)), (int(objectArray.right), int(objectArray.bottom)), color, 1)
            
            for i in range(imageH):
                for j in range(imageW):
                    if displayImasge[i][j][0]==255:
                        imageCopy[i][j]=displayImasge[i][j]
        return imageCopy
                   
            
    
    def hogFindObject(self, array, row, col, imageH, imageW):
        original = array[row][col][0]
        current = 0.00
        direction = ""
        left, top = 0
        right = imageW
        bottom = imageH
        
        while original != current:
            direction = self.HogCheckDirection(array,row,col,imageH,imageW)

            if direction == "tpl":
                current = array[row-9][col-9][0]
                if top > row:
                    top = row
                if left > col:
                    left =col
                continue

            if direction == "tp":
                current = array[row-9][col][0]
                if top > row:
                    top = row
                continue
            if direction == "tpr":
                current = array[row-9][col+9][0]
                if top > row:
                    top = row
                if right < col:
                    right =col
                continue

            if direction == "rt":
                current = array[row][col+9][0]
                if right < col:
                    right =col
                continue

            if direction == "btr":
                current = array[row+9][col+9][0]
                if bottom < row:
                    bottom = row
                if right < col:
                    right =col
                continue

            if direction == "btm":
                current = array[row+9][col][0]
                if bottom < row:
                    bottom = row
                continue

            if direction == "btl":
                current = array[row+9][col-9][0]
                if bottom < row:
                    bottom = row
                if left > col:
                    left =col
                continue

            if direction == "lft":
                current = array[row][col-9][0]
                if left > col:
                    left =col
                continue
        
        return top, left, bottom, right



    def HogCheckDirection(self, array, row, col, imageH, imageW):
        current = array[row][col][0]
        direction = ""
        tpl, tp, tpr, rt, btr, btm, btl, lft =0
        if row -9>0 and col -9>0:
            tpl = array[row -9][col-9][0]
        if row - 9>0:
            tp = array[row-9][col][0]
        if row - 9>0 and col+9<imageW:
            tpr = array[row-9][col+9][0]
        if col+9<imageW:
            rt = array[row][col+9][0]
        if row +9>imageH and col +9>imageW:
            btr = array[row -9][col-9][0]
        if row + 9<imageH:
            btm = array[row+9][col][0]
        if row + 9<imageH and col-9>0:
            btl = array[row-9][col+9][0]
        if col-9<0:
            lft = array[row][col-9][0]
        
        if tpl >= current:
            direction = "tpl"
            return direction
        if tp >= current:
            direction = "tp"
            return direction
        if tpr >= current:
            direction = "tpr"
            return direction
        if rt>= current:
            direction = "rt"
            return direction
        if btr >= current:
            direction = "btr"
            return direction
        if btm >= current:
            direction = "btm"
            return direction
        if btl >= current:
            direction = "btl"
            return direction
        if lft>= current:
            direction = "lft"
            return direction