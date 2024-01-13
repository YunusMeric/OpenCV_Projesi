import cv2
import numpy as np

def mask(matris):#Maskeleme yapıldı.
    mask = np.zeros((row,col),dtype = np.uint8)
    maske = cv2.fillPoly(mask, [matris], 255)
    result = cv2.bitwise_and(gray,gray, mask = maske)
    #cv2.imshow("Lane ",result)
    return result

def edgeDetection(result):#Kenar tespiti yapıldı.
     #kernel = np.ones((4,4),np.uint8)
     #erosion = cv2.erode(result,kernel)
     _ , thresh = cv2.threshold(result,175,255,cv2.THRESH_BINARY)
     k_algılama = cv2.Canny(thresh, 10, 100)
     #cv2.imshow("Lane Tracking",k_algılama)
     return k_algılama
        
def drawLines(k_algılama):#Şeritler çizildi ve şeritlerin ortalaması alınarak aracın anlık konumu yazdırıldı.
     saga_yatik = []
     sola_yatik = []
     #Çizgiler tespit edildi.
     lines = cv2.HoughLinesP(k_algılama, 1, np.pi/180, 1 , 1000 , 1)
     
     if not isinstance(lines, type(None)):
         for line in lines:
             for x1,y1,x2,y2 in line:
                 #Tespit edilen şeritler çizildi.
                 cv2.line(frame, (x1,y1), (x2,y2), (0,255,255),9)
                 #Şerit ortalaması için tespit edilen şeritlerin eğimi hesaplandı.
                 egim = (y2 - y1)/(x2 - x1)
                 if(x2 - x1 != 0):
                     if(egim > 0):
                         xr1, yr1, xr2, yr2 = x1, y1, x2, y2
                         saga_yatik.append((xr1, yr1, xr2, yr2))
                     else:
                         xl1, yl1, xl2, yl2 = x1, y1, x2, y2
                         sola_yatik.append((xl1, yl1, xl2, yl2))   
       #Şeritlerin orta çizgisinin konumu bulundu.
     for (xr1, yr1, xr2, yr2), (xl1, yl1, xl2, yl2) in zip(saga_yatik, sola_yatik):
         avg_x1 = (xr1 + xl1) // 2
         avg_y1 = (yr1 + yl1) // 2
         avg_x2 = (xr2 + xl2) // 2
         avg_y2 = (yr2 + yl2) // 2
         #Şeritlerin orta çizgisinin orta noktası bulundu ve aracın anlık konumu tespit edildi
         rsltx = (avg_x1 + avg_x2) // 2
         rslty = (avg_y1 + avg_y2) // 2
         coordinat = (rsltx,rslty)
         #Aracın anlık konumu circle ile gösterildi ve yazdırıldı.
         cv2.circle(frame, (avg_x1,avg_y1), 3, (0,255,0))
         cv2.putText(frame, str(coordinat),(avg_x1,avg_y1) ,cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_4)  
     cv2.imshow("Lane Tracking",frame)

# Trafik ışıkları için cascade classifier'ı oluşturuldu.
traffic_light_cascade = cv2.CascadeClassifier("haar_xml_07_19.xml")
# Araç tespiti için cascade classifier'ı oluşturuldu.
car_cascade = cv2.CascadeClassifier("cars.xml")

cam = cv2.VideoCapture("car1.mp4") 

while True:
       
  ret, frame = cam.read()
  
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  traffic_lights = traffic_light_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(30, 30))
  cars = car_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(30, 30))
  
  #Tespit edilen araçlar kare içerisine alındı ve anlık konumları üzerine yazdırıldı.       
  for (x, y, w, h) in cars:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
      coordinate = ((x + w)//2, (y + h)//2)
      cv2.putText(frame, str(coordinate),(x ,y) ,cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_4)
  
    #Tespit edilen trafik ışıkları kare içerisine alındı.
  for (x, y, w, h) in traffic_lights:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
      #Bulunan trafik ışığının konumu alınarak renk tespiti yapmak için roi oluşturuldu.
      roi = frame[y:y+h,x:x+w]
           
      bgrtohsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                
      r_lower  = np.array([0, 120, 100])
      r_upper = np.array([10, 255, 255])
      
      #Trafik ışığında renk tespit edildi.
      r_detection  = cv2.inRange(bgrtohsv, r_lower, r_upper)
      #Tespit edilen renk etrafındaki kontürler bulundu.
      contours,_ = cv2.findContours(r_detection, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      for contour in contours:
          #Tespit edilen kırmızı ışık etrafına kare çizildi ve hangi renk olduğu yazdırıldı.
          x,y,w,h = cv2.boundingRect(contour)
          cv2.rectangle(roi, (x, y), (x+w, y+h), (255, 255, 255), 1)
          #cv2.drawContours(roi,[contour], -1, (0, 255, 0), 1)
          cv2.putText(roi,"red",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1, cv2.LINE_4)
          
  
   #Şerit takip için maskeleme yapmak için çerçevenin boyutları alındı. 
  row,col = frame.shape[:2]
  #Maskeleme için kullanılan matris.(Kullandığımız videoya özgü tasarlanmıştır.)
  matris = np.array([[int((2.5*col)/8) ,int((6*row)/9)], #sol üst köşe
                     [int((5.5*col)/8) ,int((6*row)/9)] , #sağ üst köşe
                     [int((7.2*col)/8) , int((7.8*row)/9)], #sağ alt köşe
                     [int((1.5*col)/8) ,int((7.8*row)/9)]] , dtype=np.int32) #sol alt köşe
  
  drawLines(edgeDetection(mask(matris)))

  if cv2.waitKey(1) & 0xFF == ord('q'):
     break


cam.release()
cv2.destroyAllWindows()
