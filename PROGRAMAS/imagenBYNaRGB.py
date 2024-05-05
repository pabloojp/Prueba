from skimage import io, color
import matplotlib.pyplot as plt

img = io.imread("C://Users//pjime//Documents//ESTUDIOS//GRADO MATEMATICAS//4-CUARTO//TFG//A.A.A.C//FALLIDO//"
                "GITHUB_fallido//CODIGOS//CNN Microsoft//TODAS BIEN RANDOM SHUFFLE//1+27.png")
img2 = io.imread('hurricane-isabel.png ')
#imgrgb = color.gray2rgb(img)
print
plt.imshow(img)
plt.title('Original')
plt.axis('off')
plt.show()
print(img.shape)
