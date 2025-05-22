# tambem eh possivel visualizar os pgm de um jeito mais facil com a extensao PBM/PPM/PGM Viewer do VSCode
# import cv2
# cv2.imshow("Imagem", cv2.imread("./20250423/captura2/captura2_inj13_frame1.pgm", cv2.IMREAD_GRAYSCALE))
# cv2.waitKey(0)

import matplotlib.pyplot as plt
img = plt.imread('./20250423/captura1/captura1_inj6_frame15_bordas.pgm')
plt.imshow(img, cmap='gray')
plt.show()
