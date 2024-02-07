import numpy as np
import matplotlib.pyplot as plt
import torch

# fix some random seed
np.random.seed(187)
torch.manual_seed(187)

def evaluation(x, y, nn):
    m = x.shape[1]
    pred = np.zeros((1, m))
    output = nn.forward(x)

    for i in range(0, output.shape[1]):
        if output[0, i] > 0.5:
            pred[0, i] = 1
        else:
            pred[0, i] = 0

    print("Accuracy: " + str(np.sum((pred == y) / float(m))))
    return np.array(pred[0], dtype=np.int), (pred == y)[0], np.sum((pred == y) / float(m)) * 100

#
# Problem 1
#
import problem1 as p1
def problem1():

    # load data
    data= np.load('data/p1.npz')
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_test = data['X_test']
    Y_test = data['Y_test']

    print("Number of training images: %d" % X_train.shape[1])
    print("Number of test images: %d" % X_test.shape[1])

    # set network and training parameters
    net_structure = [4096, 192, 1]
    lr = 1e-4
    batch_size = 32
    num_epochs = 20

    # initialize network
    nn = p1.Network(net_structure)
    # train network
    nn.train(X_train, Y_train, lr, batch_size, num_epochs)

    # evaluate performance on the training set
    plt.figure(figsize=(10, 20))
    plt.subplots_adjust(wspace=0, hspace=0.15)
    pred_train, correctly_classified, accuracy = evaluation(X_train, Y_train, nn)
    for i in range(X_train.shape[1]):
        ax = plt.subplot(21, 10, i + 1)

        x_data = X_train[:,i].reshape(64, 64)

        if not correctly_classified[i]:
            im = ax.imshow(x_data, cmap='hot')
        else:
            im = ax.imshow(x_data, cmap='Greys_r')

        plt.xticks([])
        plt.yticks([])
        plt.suptitle(
            "Training set, number of images: %d\n Accuracy: %.2f%%, misclassified examples are represented in a red-yellow colormap."
            % (X_train.shape[1], accuracy),
            fontsize=12)


    # evaluate performance on the validation set
    plt.figure(figsize=(8, 8))
    plt.subplots_adjust(wspace=0, hspace=0.1)
    predicted, correctly_classified, accuracy = evaluation(X_test, Y_test, nn)
    for i in range(X_test.shape[1]):
        ax = plt.subplot(8, 8, i + 1)

        x_data = X_test[:,i].reshape(64, 64)

        if not correctly_classified[i]:
            im = ax.imshow(x_data, cmap='hot')
        else:
            im = ax.imshow(x_data, cmap='Greys_r')

        plt.xticks([])
        plt.yticks([])
        plt.suptitle(
            "Test set, number of images: %d\n Accuracy: %.2f%%, misclassified examples are represented in a red-yellow colormap."
            % (X_test.shape[1], accuracy),
            fontsize=12)

    plt.show()


#
# Problem 2
#
import problem2 as p2

def problem2():

    # load data
    data= np.load('data/p1.npz')
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_test = data['X_test']
    Y_test = data['Y_test']
    print("Number of training images: %d" % X_train.shape[1])
    print("Number of test images: %d" % X_test.shape[1])

    # set network and training parameters
    lr = 1e-2
    batch_size = 32
    num_epochs = 10

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = p2.CNN(num_classes=2).to(device)
    model.train()

    # setup loss, meter and optimizer
    criterion = torch.nn.CrossEntropyLoss().to(device)
    meter = p2.LossMeter()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=lr, 
                                momentum = 0.9)

    # training
    for epoch in range(1, num_epochs+1):
        # shuffle and prepare data
        imgs_train, labels_train = p1.Network.shuffle_data(None, X_train, Y_train)
        imgs_train = imgs_train.swapaxes(1,0)
        labels_train = labels_train.squeeze(0)
        # loop through data
        for train_idx in range(0, imgs_train.shape[0], batch_size):
            # batch data and move to device
            imgs_batch = imgs_train[train_idx : train_idx+batch_size].reshape(batch_size, 1, 64, 64)
            labels_batch = labels_train[train_idx : train_idx+batch_size]
            imgs_batch = torch.Tensor(imgs_batch).to(device)
            labels_batch = torch.Tensor(labels_batch).type(torch.long).to(device)

            # forward pass, backward pass and optimization
            out_train = model(imgs_batch)
            loss_train = criterion(out_train, labels_batch)
            meter.update(loss_train)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            print('Epoch: %s Step: %s Loss: %f' %(epoch, train_idx+batch_size, float(meter.get_score())))
        meter.reset()


        # evaluation loop
        model.eval()
        with torch.no_grad():
            # prep data
            img_test = torch.Tensor(X_test.swapaxes(1,0)).to(device)
            out_test = model(img_test.reshape(img_test.shape[0], 1, 64, 64))
            # get accuracy
            correctly_classified = out_test.argmax(dim=1).cpu().numpy() == Y_test
            accuracy = np.sum(correctly_classified) / float(Y_test.shape[1])
        print('Epoch: %s Accuracy: %f' %(epoch, accuracy))
        model.train()

        # evaluate performance on the validation set
        plt.figure(figsize=(8, 8))
        plt.subplots_adjust(wspace=0, hspace=0.1)
        for i in range(X_test.shape[1]):
            ax = plt.subplot(8, 8, i + 1)

            x_data = X_test[:,i].reshape(64, 64)

            if not correctly_classified.squeeze(0)[i]:
                im = ax.imshow(x_data, cmap='hot')
            else:
                im = ax.imshow(x_data, cmap='Greys_r')

            plt.xticks([])
            plt.yticks([])
            plt.suptitle(
                "Test set, number of images: %d\n Accuracy: %.2f%%, misclassified examples are represented in a red-yellow colormap."
                % (X_test.shape[1], accuracy),
                fontsize=12)
        plt.show()
         



if __name__ == "__main__":
    problem1()
    problem2()