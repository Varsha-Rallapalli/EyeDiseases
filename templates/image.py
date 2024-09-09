import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read an image
image = mpimg.imread('back1.jpg')

# Display the image
plt.imshow(image)
plt.axis('off')  # Turn off axes
plt.show()
