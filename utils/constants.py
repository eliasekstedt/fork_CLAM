"""
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
OPENAI_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_STD = [0.26862954, 0.26130258, 0.27577711]
PBO_MEAN = [0.880189, 0.703230, 0.793672]
PBO_STD = [0.123301, 0.195426, 0.128456]
PBO_SUBS_MEAN = [0.8211, 0.5220, 0.6792]
PBO_SUBS_STD = [0.1660, 0.2610, 0.1806]



MODEL2CONSTANTS = {
	"resnet50_trunc": {
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
	"uni_v1":
	{
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
	"conch_v1":
	{
		"mean": OPENAI_MEAN,
		"std": OPENAI_STD
	},
    "conch_v1_5":
    {
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
    "pbo":
    {
        "mean": PBO_MEAN,
        "std": PBO_STD
	},
    "pbo_subs":
    {
        "mean": PBO_SUBS_MEAN,
        "std": PBO_SUBS_STD,
	}
}

"""