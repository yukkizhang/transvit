from mydataset import KidneyDataset
import torch.utils.data as data
from sampler import InfiniteSamplerWrapper
import torch
import matplotlib.pyplot as plt

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

datapath = r"D:/zy/project/virtual_staining/data/kidney_trans/"
cs_dataset = KidneyDataset(datapath, train=True, transform=None)

cs_iter = iter(data.DataLoader(
    dataset=cs_dataset,
    batch_size=4,
    sampler=InfiniteSamplerWrapper(cs_dataset),
    num_workers=0
))

content_images, style_images = next(cs_iter)


# show the content and style images
# content_images_show = content_images.cpu().detach().numpy()
# style_images_show = style_images.cpu().detach().numpy()

# plt.imshow(content_images_show[0, :, :, :])
# plt.show()
# plt.imshow(style_images_show[0, :, :, :])
# plt.show()

print("finished")
