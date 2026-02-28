
"""
from torchvision import transforms

def get_eval_transforms(mean, std, target_img_size):
	trsforms = []
	trsforms.append(transforms.Resize(target_img_size))
	trsforms.append(transforms.ToTensor())
	trsforms.append(transforms.Normalize(mean, std))
	return transforms.Compose(trsforms)
"""