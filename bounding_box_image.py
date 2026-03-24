import cv2
import matplotlib.pyplot as plt

image = cv2.imread("/home/easemyai/Downloads/cat_or_dog_1.jpg")
print(image.shape)

imageRectangle = image.copy()
start_point= (300, 115)
end_point = ()
cv2.rectangle(imageRectangle, start_point, end_point, (0, 0, 255), thickness=2, lineType=cv2.LINE_8 )

cv2.imshow('original_img', image)
cv2.imshow('imageRectangle', imageRectangle)
cv2.waitKey()
cv2.destroyAllWindows()