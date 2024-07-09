import cv2
import numpy as np
import matplotlib.pyplot as plt

## hint. To install cv2, type 'pip install opencv-python'

def preprocess_image(image_path):
    # Bild laden
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Bildgröße ändern auf 28x28 Pixel
    img_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Pixelwerte normalisieren (Bereich [0, 1])
    img_normalized = img_resized / 255.0
    
    # Optional: Um das Bild an die invertierte Darstellung von MNIST anzupassen
    img_normalized = 1 - img_normalized

    # Zentrierung der Ziffer durch Auffüllen
    (thresh, img_binary) = cv2.threshold(img_resized, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)
        margin_x = (28 - w) // 2
        margin_y = (28 - h) // 2
        img_centered = np.zeros((28, 28), dtype=np.uint8)
        img_centered[margin_y:margin_y+h, margin_x:margin_x+w] = img_resized[y:y+h, x:x+w]
    else:
        img_centered = img_resized

    return img_centered

# Beispielbildpfad (hier musst du den Pfad zu deinem Bild angeben)
image_path = '.\\images\\8.jpeg'

# Bild vorverarbeiten
processed_image = preprocess_image(image_path)

# Bild anzeigen
plt.imshow(processed_image, cmap='gray')
plt.title('Processed Image')
plt.show()

# Bild speichern
plt.imsave('.\\images\\8_preprocessed.jpeg', processed_image, cmap='gray')