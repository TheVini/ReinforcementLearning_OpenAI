from scipy.ndimage.filters import gaussian_filter1d
from IPython.display import clear_output
from matplotlib import pyplot as plt
from os import walk
import numpy as np
import os
import math

output_folder = 'output_files'


def normalize(x):
    return x / np.linalg.norm(x)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def log_info(dir_path, text):
    print(text)
    with open(dir_path + '/report.txt', 'a+') as file:
        file.write(text + '\n')


def full_plot(ep_score, ep_losses, text, output_dir, batch_size=32):
    plot_score(ep_score, text, output_dir)
    plot_loss(ep_losses, text, output_dir, batch_size=batch_size)
    # plot_loss_details(ep_losses, text, output_dir)


def plot_score(ep_score, text, output_dir):
    if len(ep_score) > 0:
        clear_output(wait=True)
        ep_score = np.array(ep_score)

        ep_score_smooth = gaussian_filter1d(ep_score, sigma=4)
        fig = plt.figure(figsize=(20, 14))
        fig.suptitle(text, fontsize=14, fontweight='bold')
        plt.ylabel("Score", fontsize=22)
        plt.xlabel("Training Episodes", fontsize=22)
        plt.plot(ep_score, linewidth=0.5, color='green')
        plt.plot(ep_score_smooth, linewidth=2, color='red')
        plt.grid()
        # plt.show()
        fig.savefig(output_dir + '/score.png', bbox_inches='tight', pad_inches=0.25)
        plt.close('all')


def plot_loss(ep_losses, text, output_dir, batch_size=32):
    """
    O gráfico de losses varia muito, então para evitrar uma poluição gráfica
    é tirado uma média em batch das losses, uma média de cada batch de tamanho "batch_size"
    """
    if len(ep_losses) > batch_size:
        ep_losses_np = np.array(ep_losses)
        new_ep_loss = []

        for small_list in np.array_split(ep_losses_np, math.ceil(len(ep_losses_np)/batch_size)):
            if len(small_list) > 0:
                new_ep_loss.append(np.mean(small_list))

        ep_losses_smooth = gaussian_filter1d(new_ep_loss, sigma=4)
        fig = plt.figure(figsize=(50, 20))
        fig.suptitle(text, fontsize=15, fontweight='bold')
        plt.ylabel("Loss", fontsize=15)
        plt.xlabel("Training - Time + Episode", fontsize=15)
        plt.grid()
        plt.plot(new_ep_loss, linewidth=0.5, color='green')
        plt.plot(ep_losses_smooth, linewidth=4, color='red')
        # plt.show()
        fig.savefig(output_dir + '/loss.png', bbox_inches='tight', pad_inches=0.25)
        plt.close('all')


def disable_view_window():
    from gym.envs.classic_control import rendering
    org_constructor = rendering.Viewer.__init__

    def constructor(self, *args, **kwargs):
        org_constructor(self, *args, **kwargs)
        self.window.set_visible(visible=False)

    rendering.Viewer.__init__ = constructor


def create_dirs():
    folder_index = get_last_folder_index()
    local_output_folder = output_folder + '_' + folder_index
    local_others_dir = local_output_folder + '/others'
    local_model_output_dir = local_output_folder + '/models'
    local_video_output_dir = local_output_folder + '/videos'

    if not os.path.exists(local_output_folder):
        os.makedirs(local_output_folder)
    if not os.path.exists(local_model_output_dir):
        os.makedirs(local_model_output_dir)
    if not os.path.exists(local_video_output_dir):
        os.makedirs(local_video_output_dir)
    if not os.path.exists(local_others_dir):
        os.makedirs(local_others_dir)
    return [local_model_output_dir, local_video_output_dir, local_others_dir]


def get_last_folder_index():
    directories = []
    for (dirpath, dirnames, filenames) in walk(os.getcwd()):
        if dirpath == os.getcwd():
            [directories.append(each) for each in dirnames if each.find(output_folder) >= 0]
    directories = [each.replace(output_folder, '') for each in directories]
    directories = [each.replace('_', '') for each in directories]
    [directories.remove(each) for each in directories if len(each) == 0]
    directories = [int(each) for each in directories]
    if len(directories) == 0:
        return '001'
    return '{:03d}'.format(max(directories) + 1)
