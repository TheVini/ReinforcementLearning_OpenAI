from IPython.display import clear_output
from matplotlib import pyplot as plt
import numpy as np
import os
import shutil

others_dir = 'others'
model_output_dir = 'models'
video_output_dir = 'videos'


def normalize(x):
    return x / np.linalg.norm(x)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def full_plot(ep_score, ep_losses, text, output_dir):
    plot_score(ep_score, text, output_dir)
    plot_loss(ep_losses, text, output_dir, batch_size=32)


def plot_score(ep_score, text, output_dir):
    if len(ep_score) > 0:
        clear_output(wait=True)
        ep_score = np.array(ep_score)

        fig = plt.figure(figsize=(20, 14))
        fig.suptitle(text, fontsize=14, fontweight='bold')
        plt.ylabel("Score", fontsize=22)
        plt.xlabel("Training Epísodes", fontsize=22)
        plt.plot(ep_score, linewidth=0.5, color='green')
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
        ep_loss = []
        batches = int(len(ep_losses_np) / batch_size) if len(ep_losses_np) % batch_size == 0 else int(
            len(ep_losses_np) / batch_size) + 1
        new_ep_loss = np.array_split(ep_losses_np, batches)
        for batch in new_ep_loss:
            (ep_loss.append(np.mean(batch)) if np.mean(batch) < 10 else 10)

        fig = plt.figure(figsize=(50, 20))
        fig.suptitle(text, fontsize=15, fontweight='bold')
        plt.ylabel("Loss", fontsize=15)
        plt.xlabel("Training - Time + Episode", fontsize=15)
        plt.grid()
        plt.plot(ep_loss, linewidth=0.5, color='green')
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
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    if not os.path.exists(video_output_dir):
        os.makedirs(video_output_dir)
    if not os.path.exists(others_dir):
        os.makedirs(others_dir)
    return [model_output_dir, video_output_dir, others_dir]


def del_dirs():
    if os.path.exists(model_output_dir):
        shutil.rmtree(model_output_dir)
        print("!!! {} folder was deleted !!!".format(model_output_dir))
    if os.path.exists(video_output_dir):
        shutil.rmtree(video_output_dir)
        print("!!! {} folder was deleted !!!".format(video_output_dir))
    if os.path.exists(others_dir):
        shutil.rmtree(others_dir)
        print("!!! {} folder was deleted !!!".format(others_dir))
