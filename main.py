import cv2
import pytesseract

# Load the input image
img = cv2.imread('handwriting.jpg')

# Preprocess the image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

# Find contours and extract text regions
contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
text_regions = []
for contour in contours:
    x,y,w,h = cv2.boundingRect(contour)
    if w/h > 0.5 and w/h < 5 and w > 15 and h > 15:
        text_regions.append((x,y,w,h))

# Apply OCR to each text region and extract recognized text
recognized_text = ''
for region in text_regions:
    x,y,w,h = region
    roi = opening[y:y+h, x:x+w]
    recognized_text += pytesseract.image_to_string(roi, config='--psm 11')

# Draw rectangles around text regions and overlay recognized text on original image
for region in text_regions:
    x,y,w,h = region
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
cv2.putText(img, recognized_text, (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

# Display the result
cv2.imshow('Handwriting Recognition', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
