import matplotlib.pyplot as plt
import numpy as np


def draw_loss_plot(iter, d_a, g_a, cycle_a, d_b, g_b, cycle_b, name=''):
    plt.subplot(2, 1, 1)
    plt.plot(iter, d_a, 'r-')
    plt.plot(iter, g_a, 'g-')
    plt.plot(iter, cycle_a, 'b-')
    plt.legend(['D_A', 'G_A', 'cycle_A'])
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.subplot(2, 1, 2)
    plt.plot(iter, d_b, 'm-')
    plt.plot(iter, g_b, 'c-')
    plt.plot(iter, cycle_b, 'y-')
    plt.legend(['D_B', 'G_B', 'cycle_B'])
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.savefig('iter_based_plot_' + name + '.png')
    '''G_loss = np.array(g_a) + np.array(g_b)
    D_loss = np.array(d_a) + np.array(d_b)
    cycle_loss = np.array(cycle_a) + np.array(cycle_b)
    plt.plot(iter, G_loss, 'r-')
    plt.plot(iter, D_loss, 'g-')
    plt.plot(iter, cycle_loss, 'b-')
    plt.show()'''

def parse_train_log(file_path):
    hist = {'iter': [0], 'D_A': [0.0], 'G_A': [0.0], 'cycle_A': [0.0],
            'D_B': [0.0], 'G_B': [0.0], 'cycle_B': [0.0]}
    keys = hist.keys()
    iter = 0
    with open(file_path, 'r') as f:
        for line in f:
            if 'Training' in line:
                hist = {'iter': [0], 'D_A': [0.0], 'G_A': [0.0], 'cycle_A': [0.0],
                        'D_B': [0.0], 'G_B': [0.0], 'cycle_B': [0.0]}
                continue
            units = line.translate({ord('('): '', ord(')'): '', ord(':'): '', ord(','): ''})
            units = units.split(' ')[:-1]
            loss_dict = {}
            for i in range(0, len(units), 2):
                loss_dict[units[i]] = float(units[i + 1])
            iter += 200
            for key in keys:
                if key == 'iter': continue
                hist[key].append(loss_dict[key])
            hist['iter'].append(iter)
    for key in keys:
        hist[key] = hist[key][1:]
    return hist

hist = parse_train_log('C:\\Users\\405B\\Desktop\\pytorch_projects\\GAN\\pytorch-CycleGAN-and-pix2pix\\checkpoints\\vis2nir_aligned_with_HE_BGR2YCrCb_percep_e-2_batchnorm_cyclegan\\loss_log.txt')
draw_loss_plot(iter=hist['iter'], d_a=hist['D_A'], g_a=hist['G_A'], cycle_a=hist['cycle_A'],
               d_b=hist['D_B'], g_b=hist['G_B'], cycle_b=hist['cycle_B'], name='vis2nir_aligned_with_HE_BGR2YCrCb_percep_e-2_batchnorm')