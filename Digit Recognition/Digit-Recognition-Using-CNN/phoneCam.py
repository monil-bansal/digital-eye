import urllib
import urllib.request
import cv2
import numpy as np   
# img_resp = urllib.request.urlopen(url)
def takePhoto():
	URL = "http://192.168.0.100:8080/shot.jpg"
	with urllib.request.urlopen(URL) as url:
	    with open('temp.jpg', 'wb') as f:
	        f.write(url.read())
# img_arr = np.array(bytearray(img_resp.read()), dtype = np.uint8)
# img = cv2.imdecode(img_arr,-1)
# cv2.imshow("Phone",img)
# if cv2.waitKey(1) == 27:
#     break
    # cv2.destroyAllWindows
    