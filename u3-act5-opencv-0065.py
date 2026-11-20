# Nahum flores NC 0065
import cv2

# ==========================================
# 1. CARGAR IMAGEN ORIGINAL
# ==========================================
imagen = cv2.imread('patricio.png')
cv2.imshow('1. Original - Nahum flores NC 0065', imagen)

# ==========================================
# 2. ESCALA DE GRISES
# ==========================================
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
cv2.imwrite('patricio_gris.png', imagen_gris)
cv2.imshow('2. Escala de Grises - Nahum flores NC 0065', imagen_gris)

# ==========================================
# 3. REDIMENSIONAR (300x200)
# ==========================================
imagen_redimensionada = cv2.resize(imagen, (300, 200))
cv2.imwrite('patricio_redimensionado.png', imagen_redimensionada)
cv2.imshow('3. Redimensionada - Nahum flores NC 0065', imagen_redimensionada)

# ==========================================
# 4. ROTAR (90 GRADOS)
# ==========================================
(alto, ancho) = imagen.shape[:2]
centro = (ancho // 2, alto // 2)
matriz = cv2.getRotationMatrix2D(centro, 90, 1.0)
imagen_rotada = cv2.warpAffine(imagen, matriz, (ancho, alto))
cv2.imwrite('patricio_rotado.png', imagen_rotada)
cv2.imshow('4. Rotada - Nahum flores NC 0065', imagen_rotada)

# ==========================================
# 5. DESENFOQUE GAUSSIANO
# ==========================================
imagen_desenfocada = cv2.GaussianBlur(imagen, (15, 15), 0)
cv2.imwrite('patricio_desenfocado.png', imagen_desenfocada)
cv2.imshow('5. Desenfocada - Nahum flores NC 0065', imagen_desenfocada)

# ==========================================
# 6. DETECCIÓN DE BORDES (CANNY)
# ==========================================
bordes = cv2.Canny(imagen_gris, 100, 200)
cv2.imwrite('patricio_bordes.png', bordes)
cv2.imshow('6. Bordes Canny - Nahum flores NC 0065', bordes)

# ==========================================
# 7. CONTORNOS (Sobre copia para no alterar la original)
# ==========================================
imagen_contornos = imagen.copy()
contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imagen_contornos, contornos, -1, (0, 255, 0), 2)
cv2.imwrite('patricio_contornos.png', imagen_contornos)
cv2.imshow('7. Contornos - Nahum flores NC 0065', imagen_contornos)
# ==========================================
# ESPERAR TECLA Y CERRAR TODAS LAS VENTANAS
# ==========================================
cv2.waitKey(0)
cv2.destroyAllWindows()