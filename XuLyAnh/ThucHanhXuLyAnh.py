import sys
import tkinter
from tkinter import Frame, Tk, BOTH, Text, Menu, END
from tkinter.filedialog import Open, SaveAs
import cv2
import numpy as np
import Chapter3 as c3
import Chapter4 as c4
import Chapter9 as c9

class Main(Frame):
    
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.initUI()
  
    def initUI(self):
        self.parent.title("Digital Image Processing")
        self.pack(fill=BOTH, expand=1)
  
        menubar = Menu(self.parent)
        self.parent.config(menu=menubar)
  
        fileMenu = Menu(menubar)
        fileMenu.add_command(label="Open", command=self.onOpen)
        fileMenu.add_command(label="Save", command=self.onSave)
        fileMenu.add_separator()
        fileMenu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=fileMenu)

        chapter3Menu = Menu(menubar)
        chapter3Menu.add_command(label="Negative", command=self.onNegative)
        chapter3Menu.add_command(label="Logarit", command=self.onLogarit)
        chapter3Menu.add_command(label="Power", command=self.onPower)
        chapter3Menu.add_command(label="PiecewiseLinear", command=self.onPiecewiseLinear)
        chapter3Menu.add_command(label="Histogram", command=self.onHistogram)
        chapter3Menu.add_command(label="HistogramEqualization", command=self.onHistogramEqualization)
        chapter3Menu.add_command(label="LocalHistogram", command=self.onLocalHistogram)
        chapter3Menu.add_command(label="HistogramStatistics", command=self.onHistogramStatistics)
        chapter3Menu.add_command(label="Smoothing", command=self.onSmoothing)
        chapter3Menu.add_command(label="SmoothingGauss", command=self.onSmoothingGauss)
        chapter3Menu.add_command(label="MedianFilter", command=self.onMedianFilter)
        chapter3Menu.add_command(label="Sharpen", command=self.onSharpen)
        chapter3Menu.add_command(label="UnSharpMasking", command=self.onUnSharpMasking)
        chapter3Menu.add_command(label="Gradient", command=self.onGradient)
        menubar.add_cascade(label="Chapter3", menu=chapter3Menu)

        chapter4Menu = Menu(menubar)
        chapter4Menu.add_command(label="Spectrum", command=self.onSpectrum)
        chapter4Menu.add_command(label="FrequencyFilter", command=self.onFrequencyFilter)
        chapter4Menu.add_command(label="DrawFilter", command=self.onDrawFilter)
        chapter4Menu.add_command(label="RemoveMoire", command=self.onRemoveMoire)
        menubar.add_cascade(label="Chapter4", menu=chapter4Menu)

        chapter9Menu = Menu(menubar)
        chapter9Menu.add_command(label="Erosion", command=self.onErosion)
        chapter9Menu.add_command(label="Dilation", command=self.onDilation)
        chapter9Menu.add_command(label="OpeningClosing", command=self.onOpeningClosing)
        chapter9Menu.add_command(label="Boundary", command=self.onBoundary)
        chapter9Menu.add_command(label="HoleFill", command=self.onHoleFill)
        chapter9Menu.add_command(label="HoleFillMouse", command=self.onHoleFillMouse)
        chapter9Menu.add_command(label="MyConnectedComponent", command=self.onMyConnectedComponent)
        chapter9Menu.add_command(label="ConnectedComponent", command=self.onConnectedComponent)
        chapter9Menu.add_command(label="CountRice", command=self.onCountRice)
        menubar.add_cascade(label="Chapter9", menu=chapter9Menu)

        self.txt = Text(self)
        self.txt.pack(fill=BOTH, expand=1)
  
    def onOpen(self):
        global ftypes
        ftypes = [('Images', '*.jpg *.tif *.bmp *.gif *.png')]
        dlg = Open(self, filetypes = ftypes)
        fl = dlg.show()
  
        if fl != '':
            global imgin
            imgin = cv2.imread(fl,cv2.IMREAD_GRAYSCALE)
            # imgin = cv2.imread(fl,cv2.IMREAD_COLOR);
            cv2.namedWindow("ImageIn", cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow("ImageIn", 0, 0)
            cv2.imshow("ImageIn", imgin)

    def onSave(self):
        dlg = SaveAs(self,filetypes = ftypes);
        fl = dlg.show()
        if fl != '':
            cv2.imwrite(fl,imgout)


    def onNegative(self):
        global imgout
        imgout = np.zeros(imgin.shape, np.uint8)
        c3.Negative(imgin,imgout)
        cv2.namedWindow("ImageOut", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("ImageOut", imgout)

    def onLogarit(self):
        global imgout
        imgout = np.zeros(imgin.shape, np.uint8)
        c3.Logarit(imgin,imgout)
        cv2.namedWindow("ImageOut", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("ImageOut", imgout)

    def onPower(self):
        global imgout
        imgout = np.zeros(imgin.shape, np.uint8)
        c3.Power(imgin,imgout)
        cv2.namedWindow("ImageOut", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("ImageOut", imgout)

    def onPiecewiseLinear(self):
        global imgout
        imgout = np.zeros(imgin.shape, np.uint8)
        c3.PiecewiseLinear(imgin,imgout)
        cv2.namedWindow("ImageOut", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("ImageOut", imgout)

    def onHistogram(self):
        global imgout
        imgout = np.zeros((imgin.shape[0], 256, 3), np.uint8) + 255
        c3.Histogram(imgin,imgout)
        cv2.namedWindow("ImageOut", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("ImageOut", imgout)

    def onHistogramEqualization(self):
        global imgout
        imgout = np.zeros(imgin.shape, np.uint8)
        # c3.HistogramEqualization(imgin,imgout)
        cv2.equalizeHist(imgin, imgout)
        cv2.namedWindow("ImageOut", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("ImageOut", imgout)

    def onLocalHistogram(self):
        global imgout
        imgout = np.zeros(imgin.shape, np.uint8)
        c3.LocalHistogram(imgin, imgout)
        cv2.namedWindow("ImageOut", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("ImageOut", imgout)

    def onHistogramStatistics(self):
        global imgout
        imgout = np.zeros(imgin.shape, np.uint8)
        c3.HistogramStatistics(imgin, imgout)
        cv2.namedWindow("ImageOut", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("ImageOut", imgout)

    def onSmoothing(self):
        global imgout
        imgout = c3.Smoothing(imgin)
        cv2.namedWindow("ImageOut", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("ImageOut", imgout)

    def onSmoothingGauss(self):
        global imgout
        imgout = c3.SmoothingGauss(imgin)
        cv2.namedWindow("ImageOut", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("ImageOut", imgout)

    def onMedianFilter(self):
        global imgout
        imgout = np.zeros(imgin.shape, np.uint8)
        c3.MedianFilter(imgin, imgout)
        cv2.namedWindow("ImageOut", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("ImageOut", imgout)

    def onSharpen(self):
        global imgout
        imgout = c3.Sharpen(imgin)
        cv2.namedWindow("ImageOut", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("ImageOut", imgout)

    def onUnSharpMasking(self):
        global imgout
        imgout = c3.UnSharpMasking(imgin)
        cv2.namedWindow("ImageOut", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("ImageOut", imgout)

    def onGradient(self):
        global imgout
        imgout = c3.Gradient(imgin)
        cv2.namedWindow("ImageOut", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("ImageOut", imgout)

    def onSpectrum(self):
        global imgout
        imgout = c4.Spectrum(imgin)
        cv2.namedWindow("ImageOut", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("ImageOut", imgout)

    def onFrequencyFilter(self):
        global imgout
        imgout = c4.FrequencyFilter(imgin)
        cv2.namedWindow("ImageOut", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("ImageOut", imgout)
 
    def onDrawFilter(self):
        global imgout
        imgout = c4.DrawFilter(imgin)
        cv2.namedWindow("ImageOut", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("ImageOut", imgout)

    def onRemoveMoire(self):
        global imgout
        imgout = c4.RemoveMoire(imgin)
        cv2.namedWindow("ImageOut", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("ImageOut", imgout)

    def onErosion(self):
        global imgout
        imgout = np.zeros(imgin.shape, np.uint8)
        c9.Erosion(imgin, imgout)
        cv2.namedWindow("ImageOut", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("ImageOut", imgout)

    def onDilation(self):
        global imgout
        imgout = np.zeros(imgin.shape, np.uint8)
        c9.Dilation(imgin, imgout)
        cv2.namedWindow("ImageOut", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("ImageOut", imgout)
    def onOpeningClosing(self):
        global imgout
        imgout = np.zeros(imgin.shape, np.uint8)
        c9.OpeningClosing(imgin, imgout)
        cv2.namedWindow("ImageOut", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("ImageOut", imgout)

    def onBoundary(self):
        global imgout
        imgout = c9.Boundary(imgin)
        cv2.namedWindow("ImageOut", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("ImageOut", imgout)

    def onHoleFill(self):
        global imgout
        imgout = c9.HoleFill(imgin)
        cv2.namedWindow("ImageOut", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("ImageOut", imgout)

    def onMouse(self, event, x, y, flags, param):
        if flags & cv2.EVENT_FLAG_LBUTTON:
            cv2.floodFill(self.img, self.mask, (x,y), 255)
            cv2.imshow('ImageIn', self.img)

    def onHoleFillMouse(self):
        self.img = imgin        
        M, N = imgin.shape
        self.mask = np.zeros((M+2, N+2), np.uint8)
        cv2.setMouseCallback('ImageIn', self.onMouse)

    def onMyConnectedComponent(self):
        global imgout
        imgout = c9.MyConnectedComponent(imgin)
        cv2.namedWindow("ImageOut", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("ImageOut", imgout)

    def onConnectedComponent(self):
        global imgout
        imgout = c9.ConnectedComponent(imgin)
        cv2.namedWindow("ImageOut", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("ImageOut", imgout)

    def onCountRice(self):
        global imgout
        imgout = c9.CountRice(imgin)
        cv2.namedWindow("ImageOut", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("ImageOut", imgout)

root = Tk()
Main(root)
root.geometry("640x480+100+100")
root.mainloop()

