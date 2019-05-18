import torch
import torchvision
import os
import torch.optim as optim
from torch.autograd import Variable
import config as cf
import time
import math
import numpy as np
from transform import transform_training, transform_testing

use_cuda = torch.cuda.is_available()
decay_coefficient=2.5
best_acc = 0
lr=cf.lr


"""def adjust_learning_rate(epoch):
    learning_rate=cf.lr
    if (epoch)<= 10:
        learning_rate=cf.lr
    if (epoch) >= 10:
        learning_rate = cf.lr/ 10
        
    return learning_rate"""

def adjust_learning_rate(epoch):
    learning_rate = cf.min_learning_rate + (cf.max_learning_rate - cf.min_learning_rate) * math.exp(-epoch/cf.decay_speed)
           
    return learning_rate


def train(epoch, net, trainloader, criterion):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    lr=adjust_learning_rate(epoch)
    optimizer = optim.Adam(net.parameters(), lr, weight_decay=5e-4)
    train_loss_stacked = np.array([0])

    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, lr))
    for batch_idx, (inputs_value, targets) in enumerate(trainloader):
        if use_cuda:
            inputs_value, targets = inputs_value.cuda(), targets.cuda() # GPU settings
        optimizer.zero_grad()
        inputs_value, targets = Variable(inputs_value), Variable(targets)
        outputs = net(inputs_value)               # Forward Propagation
        loss = criterion(outputs, targets)  # Loss
        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update
        #print(batch_idx,inputs_value, targets)

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        #print(inputs_value.size(0),targets.size(0))
        correct += predicted.eq(targets.data).cpu().sum()
        train_loss_stacked = np.append(train_loss_stacked, loss.data.cpu().numpy())
    print ('| Epoch [%3d/%3d] \t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, cf.num_epochs, loss.item(), 100.*correct/total))

    return train_loss_stacked



def save_net(epoch,net):
    state = {
                'net':net.module if use_cuda else net,
                'epoch':epoch,
        }
  
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
        print('checkpoint created')
    torch.save(state, './checkpoint/savetorch.t7')
    

    
    
def test_only():
    global best_acc
    checkpoint = torch.load('./checkpoint/savetorch.t7')
    net = checkpoint['net']
    print(net)
    
    net.load_state_dict(torch.load('/Users/monster/te.pt'))
    
    visualize_model(net)    
    outputs = model(inputs)
    _, preds = torch.max(outputs.data, 1)
    
    """
    if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']
    """
    
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    sub_outputs = []
    testset =torchvision.datasets.ImageFolder(root='./Breed/Test1', transform=transform_testing())
    testloader = torch.utils.data.DataLoader(testset, batch_size=cf.batch_size, shuffle=True, num_workers=4)       

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)

        _, predicted = torch.max(outputs.data, 1)
        #print(predicted)
                    
        
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    acc = 100.*correct/total
    print("| Test Result\tAcc@1: %.2f%%" %(acc))
    if acc > best_acc:
        best_acc = acc
        #best_model_wts = model.state_dict()###
    print('* Test results : Acc@1 = %.2f%%' % (best_acc))
    #return predicted





def test_sumission(model):
    since = time.time()
    sub_outputs = []
    #net=model
    net.eval()
    ##model.train(False)  # Set model to evaluate mode
    # Iterate over data.
    for data in sub_loader:
        # get the inputs
        inputs, labels = data

        inputs = Variable(inputs.type(Tensor))
        labels = Variable(labels.type(LongTensor))

        # forward
        outputs = net(inputs)
        _, preds = torch.max(outputs.data, 1)
        sub_outputs.append(outputs.data.cpu().numpy())

    sub_outputs = np.concatenate(sub_outputs)
    for idx,row in enumerate(sub_outputs.astype(float)):
        sub_outputs[idx] = np.exp(row)/np.sum(np.exp(row))

    output_df.loc[:,1:] = sub_outputs
        
    
    time_elapsed = time.time() - since
    print('Run complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return output_df



def test(epoch, net, testloader, criterion):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_loss_stacked = np.array([0])
    ###
    #best_model_wts = model.state_dict()
    
    for batch_idx, (inputs_value, targets) in enumerate(testloader):
        if use_cuda:
            inputs_value, targets = inputs_value.cuda(), targets.cuda()
        with torch.no_grad():
            inputs_value, targets = Variable(inputs_value), Variable(targets)
        outputs = net(inputs_value)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        #print("predicted",predicted)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        test_loss_stacked = np.append(test_loss_stacked, loss.data.cpu().numpy())


    # Save checkpoint when best model
    acc = 100. * correct / total
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" % (epoch, loss.item(), acc))

    if acc > best_acc:
        best_acc = acc
        #best_model_wts = model.state_dict()###
    print('* Test results : Acc@1 = %.2f%%' % (best_acc))
    
    ###
    
    #model.load_state_dict(best_model_wts)
    #return model

    return test_loss_stacked




def start_train_test(net,trainloader, testloader, criterion):
    elapsed_time = 0
    
    for epoch in range(cf.start_epoch, cf.start_epoch + cf.num_epochs):
        start_time = time.time()

        train_loss = train(epoch, net, trainloader, criterion)
        test_loss = test(epoch, net, testloader, criterion)
        
        #save_net(epoch, net)
        torch.save(net.state_dict(), './checkpoint/savetorch_vgg.t7')

        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))

    return train_loss.tolist(), test_loss.tolist()

def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s
