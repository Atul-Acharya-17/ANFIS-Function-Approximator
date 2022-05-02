from cProfile import label
from networks.anfis import ANFIS
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from config import num_inputs, num_member_funcs, x1, x2, y, membership_funcs_params
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':

    anfis = ANFIS(num_inputs, num_member_funcs, membership_funcs_params)
    
    epochs = 100

    optimizer = torch.optim.SGD(anfis.parameters(), lr=1e-4, momentum=0.9)
    criterion = torch.nn.L1Loss(reduction='mean')

    y_true = torch.tensor(y, dtype=torch.double)

    num_data = x1.shape[0]
    for e in range(epochs):
        epoch_loss = 0
        for i in range(num_data):
            y_pred = anfis([[x1[i]], [x2[i]]])
            loss = criterion(y_pred, y_true[i])
            epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'epoch {e+1}, Loss: {epoch_loss/num_data}')

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        plt.plot(x1, x2, y, '-b', label='Training Data')
        plt.plot(x1, x2, anfis([x1, x2]).flatten().detach(), '--r', label='ANFIS Output')
        plt.title(f'Epoch {e+1}')
        plt.legend()
        plt.savefig(f"images/epoch_{e+1}.png")
        plt.close()

    print('\n')
    print('*'*100)
    print('TRAINING COMPLETE\n')
    print(anfis([x1, x2]))
    print(y_true)


    images = [Image.open(f"images/epoch_{e+1}.png") for e in range(epochs)]

    images[0].save('animation.gif', save_all=True, append_images=images[1:], duration=100, loop=0)



    
