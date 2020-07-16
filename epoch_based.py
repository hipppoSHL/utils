import matplotlib.pyplot as plt


def draw_loss_plot(epoch, d_a, g_a, cycle_a, percep_a, d_b, g_b, cycle_b, percep_b, name=''):
    plt.subplot(2, 1, 1)
    plt.plot(epoch, d_a, 'r-')
    plt.plot(epoch, g_a, 'g-')
    plt.plot(epoch, cycle_a, 'b-')
    plt.plot(epoch, percep_a, 'c')
    plt.legend(['D_A', 'G_A', 'cycle_A', 'percep_a'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.subplot(2, 1, 2)
    plt.plot(epoch, d_b, 'm-')
    plt.plot(epoch, g_b, 'c-')
    plt.plot(epoch, cycle_b, 'y-')
    plt.plot(epoch, percep_b, 'g')
    plt.legend(['D_B', 'G_B', 'cycle_B', 'percep_B'])
    plt.xlabel('Epoch')
    plt.savefig('epoch_based_plot_' + name + '.png')

def parse_train_log(file_path):
    hist = {'epoch': [0], 'D_A': [0.0], 'G_A': [0.0], 'cycle_A': [0.0], 'percep_A' : [0.0], 'D_B': [0.0], 'G_B': [0.0], 'cycle_B': [0.0], 'percep_B' : [0.0], 'cnt': [0]}
    keys = hist.keys()
    with open(file_path, 'r') as f:
        for line in f:
            if 'Training' in line:
                hist = {'epoch': [0], 'D_A': [0.0], 'G_A': [0.0], 'cycle_A': [0.0], 'percep_A': [0.0], 'D_B': [0.0],
                        'G_B': [0.0], 'cycle_B': [0.0], 'percep_B': [0.0], 'cnt': [0]}
                continue
            units = line.translate({ord('('): '', ord(')'): '', ord(':'): '', ord(','): ''})
            units = units.split(' ')[:-1]
            loss_dict = {}
            for i in range(0, len(units), 2):
                loss_dict[units[i]] = float(units[i + 1])
            epoch_cnt = int(units[1])
            if epoch_cnt not in hist['epoch']:
                hist['epoch'].append(epoch_cnt)
                for key in keys:
                    if key in ('epoch', 'cnt'): continue
                    hist[key].append(loss_dict[key])
                hist['cnt'].append(1)
            else:
                hist['cnt'][epoch_cnt] += 1
                for key in keys:
                    if key in ('epoch', 'cnt'): continue
                    hist[key][epoch_cnt] = (hist[key][epoch_cnt] * (hist['cnt'][epoch_cnt] - 1) + loss_dict[key]) / hist['cnt'][epoch_cnt]
    for key in keys:
        hist[key] = hist[key][1:]
    return hist


hist = parse_train_log('C:\\Users\\405B\\Desktop\\pytorch_projects\\GAN\\pytorch-CycleGAN-and-pix2pix\\checkpoints\\vis2nir_aligned_with_HE_BGR2YCrCb_percep_e-2_idt_cyclegan\\loss_log.txt')
draw_loss_plot(epoch=hist['epoch'], d_a=hist['D_A'], g_a=hist['G_A'], cycle_a=hist['cycle_A'],
               d_b=hist['D_B'], g_b=hist['G_B'], cycle_b=hist['cycle_B'], name='vis2nir_aligned_with_HE_BGR2YCrCb_percep_e-2_idt', percep_a=hist['percep_A'], percep_b=hist['percep_B'])
