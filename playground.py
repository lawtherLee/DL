import torch
import torch.nn as nn
import matplotlib.pyplot as plt

img = plt.imread("./data/a.jpg")
# print(f"img: {img}, shape: {img.shape}")

img2 = torch.tensor(img).permute(2, 0, 1).unsqueeze(dim=0)
# print(f"img2: {img2}, shape: {img2.shape}")

conv = nn.Conv2d(3, 4, 3)
conv_img = conv(img2)

# print(f"conv_img: {conv_img}, shape: {conv_img.shape}")

img3 = conv_img[0].permute(1, 2, 0)
# print(f"img3: {img3}, shape: {img3.shape}")

feature1 = img3[:, :, 0].detach().numpy()
plt.imshow(feature1)
plt.show()

feature1 = img3[:, :, 1].detach().numpy()
plt.imshow(feature1)
plt.show()

feature1 = img3[:, :, 2].detach().numpy()
plt.imshow(feature1)
plt.show()


feature1 = img3[:, :, 3].detach().numpy()
plt.imshow(feature1)
plt.show()
