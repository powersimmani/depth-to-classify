import torch
import torchvision
import torchvision.transforms as transforms
import code
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

def load_image():
	transform = transforms.Compose(
	    [transforms.ToTensor(),
	     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


	trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
	                                        download=False, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
	                                          shuffle=True, num_workers=2)

	testset = torchvision.datasets.CIFAR10(root='./data', train=False,
	                                       download=False, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=4,
	                                         shuffle=False, num_workers=2)

	classes = ('plane', 'car', 'bird', 'cat',
	           'deer', '-dog', 'frog', 'horse', 'ship', 'truck')

	# functions to show an image

	def imshow(img):
	    img = img / 2 + 0.5     # unnormalize
	    npimg = img.numpy()
	    plt.imshow(np.transpose(npimg, (1, 2, 0)))
	    plt.savefig('a.png')


	# get some random training images
	dataiter = iter(trainloader)
	images, labels = dataiter.next()

	# show images
	imshow(torchvision.utils.make_grid(images))
	# print labels
	print(' '.join('%5s' % classes[labels[j]] for j in range(4)))           
	code.interact(local=dict(globals(), **locals()))

def 

if __name__ == "__main__":
	load_image()