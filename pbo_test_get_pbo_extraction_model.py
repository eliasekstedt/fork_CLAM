import torchvision
import torch

def get_pbo_encoder(path_model):
    """
    this function is code modified from:
    https://github.com/ozanciga/self-supervised-histopathology?tab=readme-ov-file
    the .ckpt file can also be downloaded from there.
    """
    def load_model_weights(model, weights):
        model_dict = model.state_dict()
        weights = {k: v for k, v in weights.items() if k in model_dict}
        if weights == {}:
            print('No weight could be loaded..')
        model_dict.update(weights)
        model.load_state_dict(model_dict)
        return model

    model = torchvision.models.__dict__['resnet18'](weights=None)
    state = torch.load(path_model, map_location='cuda:0', weights_only=False)
    state_dict = state['state_dict']
    for key in list(state_dict.keys()):
        state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
    model = load_model_weights(model, state_dict)
    model.fc = torch.nn.Sequential()
    #model = model.cuda()
    return model

path_model = 'pbo_model/pbo_res18.ckpt'
model = get_pbo_encoder(path_model)
model = model.cuda()
images = torch.rand((10, 3, 224, 224), device='cuda')

out = model(images)
print(out.shape)