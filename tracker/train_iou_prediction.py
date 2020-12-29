"""
Trains a classifier to predict the IoU of a randomly generated bounding box.

"""
import torch
from torch.utils.data import Dataset
from torchvision import datasets, models
from IPython import embed
import os
import matplotlib.pyplot as plt
import utils_2
import random
import argparse
import wandb
import numpy as np

# This was written with a different configuration than I typically use.
# And I have no qualms about disabling no-member
# pylint: disable=bad-indentation, no-member
class EarlyStopper():
    """Performs early stopping.

    Attributes:
        best_loss: the best loss seen so far.
        counter: how many epochs since the best loss was reset.
        output_dir: where to save the model.
        patience: how long to wait before exiting.
    """
    def __init__(self, output_dir, patience):
        """Initialize the early stopping module

        args:
            output_dir: where to save best loss checkpoints.
            patience: how long to wait before exiting.
        """

        self.output_dir = output_dir
        self.patience = patience

        self.counter = 0
        self.best_loss = 1e32

    def step(self, model, loss):
        """Performs an early stopping step.

        args:
            model: the model to save if we have the best loss
            loss: the val loss of the most recent run.

        returns:
            A boolean representing whether or not to terminate the run.
        """
        if loss < self.best_loss:
            self.best_loss = loss
            torch.save(
                model.state_dict(), self.output_dir +\
                "/checkpoint_best_loss.weights")
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True

        return False

class CocoDataset(Dataset):
    """Loads the MSCOCO Dataset.

    Attributes:
        idx_list: matches an image-object index to a single index.
        dataset: the CocoDetection dataset.
        to_tensor: transforms an image to a tensor.
    """
    def __init__(self, image_dir, json):
        """Initialize the mscoco dataset.

        Args:
            image_dir: The directory where the MSCOCO images are stored.
            json: The MSCOCO json file.
        """
        self.idx_list = []
        self.dataset = datasets.CocoDetection(image_dir, json)

        # Create the mapping for a single list item to correspond to
        # a single object.
        for i in range(len(self.dataset)):
            for j in range(len(self.dataset[i][1])):
                if self.dataset[i][1][j]['bbox'][2] < 0.1 or\
                   self.dataset[i][1][j]['bbox'][3] < 0.1:
                    print("Found bbox with area 0")
                    continue
                self.idx_list.append((i, j))
                # Stop at 50,000 objects.
                if len(self.idx_list) > 50000:
                    return

    def __len__(self):
        """Returns the length of the dataset.

        Returns:
            Dataset length
        """
        return len(self.idx_list)

    def __getitem__(self, idx):
        """Retrieves a sample from the dataset.

        Args:
            idx: The index of the desired sample.

        Returns:
            A dict containing the transformed image and its iou with the
            gold-standard bbox.
        """

        # Extract data for clarity
        indices = self.idx_list[idx]
        first_layer = self.dataset[indices[0]]
        image = first_layer[0]
        value = first_layer[1][indices[1]]

        base_bbox = value['bbox']

        # mscoco gives tlx tly w h, but needs to be split for my functions.
        target_pos = torch.tensor([base_bbox[0], base_bbox[1]]).float()
        target_sz = torch.tensor([base_bbox[2], base_bbox[3]]).float()

        # I believe this was made unnecessary via checks in init, but I'll
        # leave it in just in case.
        if target_sz[0] == 0 or target_sz[1] == 0:
            print("Found bbox with area 0 in getitem")
            embed()

        # Create the jittered bounding box to match a maximum allowable
        # IoU.
        allowable_iou = random.random()
        jitter_bbox_pos, jitter_bbox_sz = utils_2.generate_random_bbox(
            target_pos, target_sz, 1, allowable_iou, True)
        jitter_bbox_pos = jitter_bbox_pos.squeeze()
        jitter_bbox_sz = jitter_bbox_sz.squeeze()
        # bounding box is returned tlx tly, w, h
        modified_image = utils_2.add_fourth_channel(
            image, jitter_bbox_pos, jitter_bbox_sz)

        # if the jittered bbox is greater than or less than image size,
        # truncate it.
        if jitter_bbox_pos[0] < 0:
            jitter_bbox_pos[0] = torch.tensor(0).float()
        if jitter_bbox_pos[1] < 0:
            jitter_bbox_pos[1] = torch.tensor(0).float()
        if jitter_bbox_pos[0] + jitter_bbox_sz[0] > image.size[0]:
            jitter_bbox_sz[0] = torch.tensor(
                image.size[0] - jitter_bbox_pos[0]).float()
        if jitter_bbox_pos[1] + jitter_bbox_sz[1] > image.size[1]:
            jitter_bbox_sz[1] = torch.tensor(
                image.size[1] - jitter_bbox_pos[1]).float()
        # and calculate the IoU. Because of the above does not necessarily
        # correspond to the allowable iou
        iou = utils_2.calc_iou(
            torch.tensor(
                [[target_pos[0], target_pos[1], target_sz[0], target_sz[1]]]),\
            torch.tensor(
                [[jitter_bbox_pos[0], jitter_bbox_pos[1], jitter_bbox_sz[0], \
                  jitter_bbox_sz[1]]]))

        return {'image': modified_image, 'iou': iou}

def train_step(dataset, network, optimizer):
    """Performs a training step.

    Args:
        dataset: The training dataset.
        network: The network to train.
        optimizer: The optimizer.

    Returns:
        The mean training loss for the epoch.
    """

    loss_fn = torch.nn.CrossEntropyLoss()
    all_mean_losses = []
    network.train()
    for _, information_dict in enumerate(dataset):
        network_out = network(information_dict['image'].cuda())[:, :10]

        # Our target is the a binned value that most closely matches the
        # true IoU.
        target = (information_dict['iou'] * 10).floor().long().view(-1)
        # round down if the target IoU is 1
        target = (target >= 10).long()*9*torch.ones(target.shape).long() +\
                (target < 10).long()*(target >= 0).long()*target +\
                (target < 0).long()*torch.zeros(target.shape).long()
        try:
            loss = loss_fn(network_out, target.cuda())
        except:
            print(target.min(), target.max())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_mean_losses.append(loss.item())

    return {'train/mean_loss': np.array(all_mean_losses).mean()}

def eval_step(dataset, network):
    """Runs an evaluation step.

    Args:
        dataset: The evaluation dataset.
        network: The network to evaluate.

    Returns:
        A dict containing the mean loss and 0-1 accuracy.
    """
    network.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    all_mean_losses = []
    correct_guesses = 0
    total_guess = 0
    for _, information_dict in enumerate(dataset):
        network_out = network(information_dict['image'].cuda())[:, :10]
        # Our target is the a binned value that most closely matches the
        # true IoU.
        target = ((information_dict['iou']) * 10).floor().long().view(-1)

        # round down if the target IoU is 1
        target = (target >= 10).long()*9*torch.ones(target.shape).long()\
                + (target < 10).long()*(target >= 0).long()*target\
                + (target < 0).long()*torch.zeros(target.shape).long()
        loss = loss_fn(network_out, target.cuda())
        try:
            all_mean_losses.append(loss.item())
        except:
            print(target.min(), target.max())
        correct_guesses += (
            network_out.argmax(dim=1).cpu() == target.cpu()).float().sum()
        total_guess += network_out.shape[0]

    to_return = {}
    to_return['eval/mean_loss'] = np.array(all_mean_losses).mean()
    to_return['eval/acc'] = correct_guesses/total_guess

    return to_return

PARSER = argparse.ArgumentParser(description="Train IoU predictor on MSCOCO.")
PARSER.add_argument("--lr", type=float, default=1e-4, help="Learning Rate")
PARSER.add_argument("--batch_size", type=int, default=16, help="Batch size.")
PARSER.add_argument("--num_epochs", type=int, default=100,
                    help="How many epochs to train for.")
PARSER.add_argument("--eval_step", type=int, default=1,
                    help="How often to perform an eval step.")
PARSER.add_argument("--patience", type=int, default=10,
                    help="The patience for the early stopper.")
PARSER.add_argument("--output_dir", type=str, default="output",
                    help="The base folder to save in.")
PARSER.add_argument("--arch", type=str, default="resnet-18",
                    help="Which architecture to use.")
ARGS = PARSER.parse_args()

# Switch the backend so that we can run headless.
plt.switch_backend('Agg')

# Initialize the weights and biases run
wandb.init(project="iou_regressor", config=ARGS.__dict__)

# Load the dataset.
coco_train = CocoDataset('/z/dat/mscoco/images/train2014',
                         '/z/dat/mscoco/annotations/instances_train2014.json')
coco_val = CocoDataset('/z/dat/mscoco/images/val2014',
                       '/z/dat/mscoco/annotations/instances_val2014.json')

# Create the output directory.
try:
    OUTPUT_DIR = os.path.join(
        ARGS.output_dir, wandb.run.project, wandb.sweep.id, wandb.run.id)
except AttributeError:
    OUTPUT_DIR = os.path.join(
        ARGS.output_dir, wandb.run.project, wandb.run.id)

os.makedirs(OUTPUT_DIR)

# Initialize the dataloaders.
train_loader = torch.utils.data.DataLoader(
    coco_train, batch_size=ARGS.batch_size, num_workers=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(
    coco_val, batch_size=ARGS.batch_size, num_workers=8, shuffle=False)

# Use the preloaded resnet model.
if ARGS.arch == "resnet-18":
    NETWORK = models.resnet18(pretrained=True).cuda()
if ARGS.arch == "resnet-34":
    NETWORK = models.resnet18(pretrained=True).cuda()
if ARGS.arch == "resnet-50":
    NETWORK = models.resnet18(pretrained=True).cuda()
if ARGS.arch == "resnet-101":
    NETWORK = models.resnet18(pretrained=True).cuda()
if ARGS.arch == "resnet-152":
    NETWORK = models.resnet18(pretrained=True).cuda()

# Add the fourth channel.
NETWORK.conv1.weight = torch.nn.Parameter(
    torch.cat((NETWORK.conv1.weight.detach(),
               torch.rand(64, 1, 7, 7).cuda()), dim=1))

# Create the optimizer
OPTIMIZER = torch.optim.Adam(NETWORK.parameters(), lr=ARGS.lr)

# Create the early stopping module.
early_stopper = EarlyStopper(OUTPUT_DIR, ARGS.patience)

# Loop through all of the epochs.
for cur_epoch in range(ARGS.num_epochs):
    print("Epoch " + str(cur_epoch))

    # If it's time to evaluate, evaluate.
    if cur_epoch % ARGS.eval_step == 0:
        with torch.no_grad():
            eval_to_log = eval_step(val_loader, NETWORK)
        eval_to_log['epoch'] = cur_epoch
        # Log the output of the eval step to wandb.
        wandb.log(eval_to_log)
        # Check early stopping.
        if early_stopper.step(NETWORK, eval_to_log['eval/mean_loss']):
            break

    # Perform a train step, checkpoint every epoch, log to wandb.
    train_to_log = train_step(train_loader, NETWORK, OPTIMIZER)
    train_to_log['epoch'] = cur_epoch
    torch.save(
        NETWORK.state_dict(), OUTPUT_DIR +\
        "/checkpoint_last_epoch.weights")
    wandb.log(train_to_log)
