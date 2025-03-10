from resnet import ResNet9
from vgg import VGG

def get_model(data, args):
    if data == 'cifar10':
        model = ResNet9(3,num_classes=10, args=args)
    elif data == 'cifar100':
        model = VGG('VGG9',num_classes=100)
    return model
         