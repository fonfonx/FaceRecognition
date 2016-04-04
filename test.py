from cv2.cv import *

img = LoadImage("../AR_crop/M-002-01.bmp")
NamedWindow("opencv")
ShowImage("opencv",img)
WaitKey(0)
