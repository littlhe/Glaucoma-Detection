import cv2
import glob

path = "E:/Drishti/smooth/*.*"
img_number = 1

for file in glob.glob(path):
    print(file)
    img = cv2.imread(file, 1)


    #Converting image to LAB Color so CLAHE can be applied to the luminance channel
    lab_img= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    #Splitting the LAB image to L, A and B channels, respectively
    l, a, b = cv2.split(lab_img)



    #Histogram Equlization
    #Apply histogram equalization to the L channel
    #equ = cv2.equalizeHist(l)

    #Combine the Hist. equalized L-channel back with A and B channels
    #updated_lab_img1 = cv2.merge((equ,a,b))

    #Convert LAB image back to color (RGB)
    #hist_eq_img = cv2.cvtColor(updated_lab_img1, cv2.COLOR_LAB2BGR)

    #cv2.imwrite("E:\Drishti\histoeq\histoeq_image"+str(img_number)+".png", hist_eq_img)
    #img_number +=1
    
    
    #CLAHE
    #Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(l)

    #Combine the CLAHE enhanced L-channel back with A and B channels
    updated_lab_img2 = cv2.merge((clahe_img,a,b))

    #Convert LAB image back to color (RGB)
    CLAHE_img = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)

    cv2.imwrite("E:/Drishti/smooth_clahe/smooth_clahe_image"+str(img_number)+".png", CLAHE_img)
    img_number +=1
