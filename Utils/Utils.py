from scipy.ndimage.filters import gaussian_filter1d
from IPython.display import clear_output
from skimage.transform import resize
from matplotlib import pyplot as plt
from PIL import Image
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


def log_info(dir_path, text, file='report.txt'):
    print(text)
    with open(dir_path + '/' + file, 'a+') as file:
        file.write(text + '\n')


def test_log_info(dir_path, text, file_number=None, file_name='test_report.txt'):
    if file_number is not None:
        file_name = 'test_report_{}.txt'.format(file_number)
    log_info(dir_path, text, file=file_name)
    

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
        with open(output_dir + '/losses.txt', 'w+') as file:
            for each in new_ep_loss:
                file.write(str(each)+'\n')


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


def downscale_obs(obs, new_size=(42, 42), to_gray=True):
    """
    Downscale_obs: rescales RGB image to lower dimensions with option to change to grayscale
    obs: Numpy array or PyTorch Tensor of dimensions Ht x Wt x 3 (channels)
    to_gray: if True, will use max to sum along channel dimension for greatest contrast
    """
    if to_gray:
        return resize(obs, new_size, anti_aliasing=True).max(axis=2)
    else:
        return resize(obs, new_size, anti_aliasing=True)


def cut_image(state):
    """
    Remove the upper image part that contains score  info, lifes, world and time.
    :param state: current state
    :return:
    """
    PIL_image = Image.fromarray(state.astype('uint8'), 'RGB')
    return np.array(PIL_image.crop((0, 32, 256, 240)))


def preprocess_state(state_to_process, new_state_size):
    """
    Reduce image scale, reshape, and reduce image pixels values
    :param state_to_process: new state
    :param new_state_size: target state
    :return: preprocessed state
    """
    state_to_process = downscale_obs(state_to_process, new_state_size)
    state_to_process = state_to_process.reshape([1, new_state_size[0], new_state_size[1], 1]).astype('float32')
    return state_to_process / 255.


def prepare_initial_state(state, state_size, channels=3):
    """
    Preprocess image to reduce state and channels, and cut image
    :param state: current state
    :param state_size: target state size
    :param channels: target channels
    :return:
    """
    state = cut_image(state)
    state = preprocess_state(state, state_size)
    return state.repeat(channels, axis=channels)


def prepare_multi_state(new_state, state_size, old_state, channels=3, state_single_size=42):
    """
    Create a sequence of "channels" different images.
    :param new_state: current state to process
    :param state_size: target state size
    :param old_state: old frames
    :param channels: target channels amount
    :param state_single_size: target state size for channels sequence
    :return: a sequence of channels with different images
    """
    new_state = cut_image(new_state)
    old_state = np.reshape(old_state, (1, channels, state_single_size, state_single_size))
    new_state = np.reshape(preprocess_state(new_state, state_size),
                           (1, 1, state_single_size, state_single_size))

    for i in range(channels):
        if i == (channels - 1):
            old_state[0][i] = new_state
        else:
            old_state[0][i] = old_state[0][i + 1]

    return np.reshape(old_state, (1, state_single_size, state_single_size, channels))