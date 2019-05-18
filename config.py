start_epoch = 1
num_epochs = 150
batch_size = 256
optim_type = 'Adam'
#resize=256
resize=224
crop_size=224
lr=0.01
decay_coefficient=2.5

max_learning_rate = 0.005
min_learning_rate = 0.0002
#120 = number of classes
decay_speed = decay_coefficient*120/batch_size


mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
    'mnist': (0.1307,),
    'stl10': (0.485, 0.456, 0.406),
}

std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
    'mnist': (0.3081,),
    'stl10': (0.229, 0.224, 0.225),
}
