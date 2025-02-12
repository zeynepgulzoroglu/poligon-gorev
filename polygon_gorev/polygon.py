import cv2
import os
import numpy as np

model_psth= os.path.join(os.getcwd())
print(model_psth)
folder = "images"
images = [f for f in os.listdir(folder) if f.endswith('.png')]

intervals = [((407, 434), (407, 434)), 
             ((430, 460), (180, 190)), 
             ((761, 479), (761, 479)), 
             ((260, 281), (350, 370)), 
             ((123, 395), (111, 510))]
values = [1, 2, 3, 4, 5]

first_image = os.path.join(folder, images[0])
previous_img = cv2.imread(first_image)

total_score_cumulative = 0

for i in range(0, len(images)):  
    image_path = os.path.join(folder, images[i])
    img = cv2.imread(image_path)

    shots = cv2.absdiff(previous_img, img)
    gray = cv2.cvtColor(shots, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow("abs",thresh)
    total_score_current = 0

    for contour in contours:
        x, y, _, _ = cv2.boundingRect(contour)

        for j, ((x1, x2), (y1, y2)) in enumerate(intervals):
            if x1 <= x < x2 and y1 <= y < y2:
                assigned_value = values[j]
                total_score_current += assigned_value
                break

    total_score_cumulative += total_score_current
    previous_img = img 

    cv2.putText(img, f"Toplam Puan: {total_score_cumulative}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    
    cv2.imshow(f"{images[i]}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()