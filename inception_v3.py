'''
Inception_V3 model for image embedding

Author:
	Yuchen (Peter) Ge

E-mail:
	yge15@jhmi.edu

Usage:
	$ from inception_v3 import new_inception_v3 as icpv3

Attributes:
    set_parameter_requires_grad():  freeze all parameters to extract features
    new_inception_v3():             customize num of classes after image embeddings       
	
'''
#%%
import torchvision.models as models
import torch.nn as nn
#%%
def set_parameter_requires_grad(model, feature_extracting):
    '''
	Set trainbale parameters. 
    For feature extraction, freeze all parameters but those in the last layer.

	Args:
		model (inception_v3): 		    the pre-trained inception v3 model
		feature_extracting (bool): 		set true to freeze all paraters

	Returns:
		None

	'''
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
#%%
def new_inception_v3(num_classes):
    '''
    Create our own inception model. 

	Args:
		num_classes (int): 		        number of classes in the output

	Returns:
        inception (inception_v3):       model with replaced fully connected layers and frozen parameters

    '''
    inception = models.inception_v3(pretrained=True)
    set_parameter_requires_grad(inception, True)

    inception.AuxLogits.fc = nn.Linear(768, num_classes)
    inception.fc = nn.Linear(2048, num_classes)

    return inception
#%%
# Find total parameters and trainable parameters
net = new_inception_v3(100) # (2048+768)*100 + 100+100 = 281,800
total_params = sum(p.numel() for p in net.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')