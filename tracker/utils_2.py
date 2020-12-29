"""Util functions for seed rejection on tracking
"""
import torch
from IPython import embed
from torchvision import transforms

def add_fourth_channel(image, random_pos, random_sz):
    """ Performs normalization and adds fourth channel.

    args:
        image: The three channel RGB image.
        random_pos: the tlx tly of the bounding box
        random_sz: the width height of the bounding box.

    returns:
        a four channel tensor where the bounding box is the fourth channel.
    """
    to_tensor = transforms.ToTensor()
    image_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_tensor = to_tensor(image)
    image_tensor_transformed = image_transform(image_tensor)
    return_tensor = torch.zeros(4, 960, 960)
    return_tensor[:3, 0:image_tensor.shape[1], 0:image_tensor.shape[2]] = image_tensor_transformed

    x_range_tensor = torch.arange(960).view(1, -1).repeat(960, 1)
    y_range_tensor = torch.arange(960).view(-1, 1).repeat(1, 960)

    # Expand if it's not batched.
    # TODO: Batching is inconsistent. Images are dim 3.
    if len(random_pos.shape) == 1:
        random_pos = random_pos.unsqueeze(0)
        random_sz = random_sz.unsqueeze(0)

    random_bbox_corners = [
        random_pos[:, 0],
        random_pos[:, 1],
        random_pos[:, 0] + random_sz[:, 0],
        random_pos[:, 1] + random_sz[:, 1]]

    random_bbox_tensor = torch.tensor(random_bbox_corners)

    # the mask is true if:
    #    the value of x is greater than the minimum bbox x AND
    #    the value of y is greater than the minimum bbox y AND
    #    the value of x is less than the maximum bbox x AND
    #    the value of y is less than the maximum bbyx y AND
    #    the value of x is less than the image width AND
    #    the value of y is less than the image height
    # the minimum x and y are implicit in the creation of the mask.
    mask = (x_range_tensor > random_bbox_tensor[0]) * (y_range_tensor > random_bbox_tensor[1]) * (x_range_tensor < random_bbox_tensor[2]) * (y_range_tensor < random_bbox_tensor[3]) * (x_range_tensor <= image_tensor.shape[1]) * (y_range_tensor <= image_tensor.shape[2])
    mask = mask.float()

    return_tensor[3, :, :] = mask

    # uncomment to visually inspect bboxes/masks
    reversed_return_tensor = torch.zeros(return_tensor.shape)
    reversed_return_tensor[:3, :image_tensor.shape[1], :image_tensor.shape[2]] = image_tensor
    reversed_return_tensor[3, :, :] = mask

    to_pil = transforms.ToPILImage()
    show_image = to_pil(reversed_return_tensor)

    return return_tensor

def generate_random_bbox(target_pos, target_sz, num_samples, min_iou=0.8, hard_min=False):
    """ Generates a random bounding box with some minimum overlap with the
    ground truth box.

    We note that the bounding box may run off the edge of the image. While
    this is not ideal, it's also something that would require more
    restructuring than we're willing to do. This will not decrease IoU.

    Args:
        target_pos: the tlx tly position of the original bounding box
        target_sz: the size of the original bounding box
        count: how many random bboxes do we want?
        min_iou: the minimum overlap.

    Returns:
        A batch of count bounding boxes in the form of a target position
        (tlx, tly) and size.
    """

    # I had written this once already, and had to rewrite it because it
    # wasn't working, and the code was too messy to debug. So I'm probably
    # going to overcompensate. Test with test_random_bbox_gen.py

    # The first step is to find the random offsets, which we do in one step
    # (as opposed to drawing randomly when they're needed). I don't know if
    # this is faster, but it makes things tidier.

    # The high-level strategy is to first generate a bounding box which meets
    # the criteria when compared to a unit (width one, height one, center
    # (0.5, 0.5)). This simplifies the calculations, since we can pull out
    # zeros, and the corners can be treated as constants.

    # we begin with the IoU formula:
    # iou = (min(BRX_1, BRX_2)-max(TLX_1, TLX_2))*(
    # min(BRY_1, BRY_2)-max(TLY_1, TLY_2))
    # which simplifies for the unit bbox to:
    # iou = (min(BRX, 1)-max(TLX, 0))*(min(BRY, 1)-max(TLX, 0))

    # We then select one corner at a time in order TLX, BRX, TLY, BRY,
    # and randomly select it based on bounds set by the previous corners.
    # Future corners are set to their ideal value (either 0 or 1).

    # The random offsets applied to the min/max vals
    shifts = torch.rand(num_samples, 4)

    # The way the equation is set up strongly biases towards the minimum IoU.
    # So we create a new min_IoU between the current IoU and one.
    if hard_min == False:
        min_iou = (1 - min_iou)*torch.rand(1).item()+min_iou

    # Start with tlx
    tlx_min = 1. - 1./min_iou # tlx < 0
    tlx_max = 1-min_iou # tlx > 0
    tlx = (tlx_max-tlx_min) * shifts[:, 0] + tlx_min

    # max_tlx is used to keep notation cleaner, corresponds to max(TLX, 0)
    max_tlx = (tlx > 0)*tlx

    # brx now
    brx_min = min_iou*(1-tlx+max_tlx)+max_tlx # brx < 1
    brx_max = (1-max_tlx)/min_iou + tlx - max_tlx # brx > 1
    brx = (brx_max-brx_min) * shifts[:, 1] + brx_min

    # similar to max_tlx, min_brx corresponds to min(BRX, 1)
    min_brx = (brx < 1)*brx + (brx >= 1)*torch.ones(brx.shape)

    # and for neatness we keep track of the two differences in x, with
    # and without the max/min operations.
    int_x_diff = min_brx - max_tlx
    union_x_diff = brx - tlx

    # tly
    tly_max = -1/((int_x_diff/min_iou)-union_x_diff+int_x_diff)+1 # tly < 0
    tly_min = -1/union_x_diff * (
        int_x_diff/min_iou-union_x_diff-1+int_x_diff) # tly > 0
    tly = (tly_max-tly_min) * shifts[:, 2] + tly_min

    # equivalent to max_tlx
    max_tly = (tly > 0) * tly

    # BRY
    # I solved the previous by hand. I solved these by matlab.
    bry_min = (
        min_iou + max_tly*int_x_diff - min_iou*tly*union_x_diff +
        min_iou*max_tly*int_x_diff)/(int_x_diff - min_iou*union_x_diff +
                                     min_iou*int_x_diff)
    bry_max = 1/union_x_diff * ((
        int_x_diff-int_x_diff*max_tly)/min_iou+tly*union_x_diff -\
        int_x_diff*(max_tly-1)-1)

    # Use the greater than to make sure it's pinned to min (target) iou
    bry = (bry_max-bry_min) * (shifts[:, 3] > 0.5).float() + bry_min

    # place in image coordinates
    # We create the transform required to move the unit square to the
    # target bounding box. 

    # we first scale
    tlx_scaled = tlx * target_sz[0]
    tly_scaled = tly * target_sz[1]

    brx_scaled = brx * target_sz[0]
    bry_scaled = bry * target_sz[1]

    # then we shift
    tlx_im_coords = tlx_scaled + target_pos[0]
    tly_im_coords = tly_scaled + target_pos[1]
    brx_im_coords = brx_scaled + target_pos[0]
    bry_im_coords = bry_scaled + target_pos[1]

    # Convert to tlx tly width height
    width = brx_im_coords - tlx_im_coords
    height = bry_im_coords - tly_im_coords

    target_pos_to_return = torch.zeros(num_samples, 2)
    target_sz_to_return = torch.zeros(num_samples, 2)

    target_pos_to_return[:, 0] = tlx_im_coords
    target_pos_to_return[:, 1] = tly_im_coords

    target_sz_to_return[:, 0] = width
    target_sz_to_return[:, 1] = height

    return target_pos_to_return, target_sz_to_return

def intersect(box_p, box_t):
    """Calculates the intersect between two bounding boxes.

    args:
        box_p: The proposed box.
        box_t: The target box.

    returns:
        the area of the intersection between the two bboxes.
    """
    x_left = torch.max(box_p[:, 0], box_t[:, 0])
    y_top = torch.max(box_p[:, 1], box_t[:, 1])
    x_right = torch.min(box_p[:, 2], box_t[:, 2])
    y_bottom = torch.min(box_p[:, 3], box_t[:, 3])

    width = torch.clamp(x_right-x_left, min=0)
    height = torch.clamp(y_bottom-y_top, min=0)

    intersect_area = width*height

    return intersect_area

def calc_iou(guess, target):
    """Calculate the iou across a batch of bboxes.

    args:
        guess: the batch of guess bounding boxes (tlx tly, w h)
        target: the batch of target bounding boxes. (tlx tly, w h)

    returns:
        intersection over union of every guess-target pair.
    """
    # change from tlx/tly/width/height to corners
    box_p = torch.cat([(guess[:, 0]).unsqueeze(0),\
                       (guess[:, 1]).unsqueeze(0),\
                       (guess[:, 0]+guess[:, 2]).unsqueeze(0),\
                       (guess[:, 1]+guess[:, 3]).unsqueeze(0)])
    # switching from 4xbatch to batchx4
    box_p = box_p.transpose(0, 1)

    # and repeat for target
    box_t = torch.cat([(target[:, 0]).unsqueeze(0),\
                       (target[:, 1]).unsqueeze(0),\
                       (target[:, 0]+target[:, 2]).unsqueeze(0),\
                       (target[:, 1]+target[:, 3]).unsqueeze(0)])
    box_t = box_t.transpose(0, 1)

    intersect_area = intersect(box_p, box_t)

    box_p_area = (box_p[:, 2] - box_p[:, 0])*(box_p[:, 3]-box_p[:, 1])
    box_t_area = (box_t[:, 2] - box_t[:, 0])*(box_t[:, 3]-box_t[:, 1])

    union = box_p_area + box_t_area - intersect_area
    overlap = intersect_area/union

    if torch.isnan(overlap.sum()):
        from IPython import embed
        print("NaN overlap")
        embed()
    assert (overlap >= 0).all()
    assert (overlap <= 1).all()

    return overlap

def calc_iou_with_mean(region_list):
    """ Finds the mean track, then calculates the mean absolute IoU.

    args:
        region_list: a list of tracks
        (each of which is a set of tlx-tly-w-h bboxes)

    Returns the mean absolute iou of all frames with the mean track.
    """
    # Convert the region list into batch x frames x coords to make it simpler
    num_trials = len(region_list)
    num_frames = len(region_list[0])

    tensorized = torch.zeros((num_trials, num_frames, 4))

    trial_count = 0
    for cur_trial in region_list:
        frame_count = 0
        for cur_frame in cur_trial:
            tensorized[trial_count, frame_count, :] = torch.tensor(cur_frame)
            frame_count += 1
        trial_count += 1

    # convert from w/h to brx bry.
    tensorized[:, :, 2] = tensorized[:, :, 0] + tensorized[:, :, 2]
    tensorized[:, :, 3] = tensorized[:, :, 1] + tensorized[:, :, 3]

    frame_means = tensorized.mean(dim=0).unsqueeze(0).repeat(num_trials, 1, 1).unsqueeze(0)
    tensorized = tensorized.unsqueeze(0)
    for_iou = torch.cat((frame_means, tensorized), dim=0) 

    # We can now easily find the corners of our intersect bbox
    intersect_tlx = torch.max(for_iou[:, :, :, 0], dim=0)[0]
    intersect_tly = torch.max(for_iou[:, :, :, 1], dim=0)[0]
    intersect_brx = torch.min(for_iou[:, :, :, 2], dim=0)[0]
    intersect_bry = torch.min(for_iou[:, :, :, 3], dim=0)[0]

    intersect_area = (intersect_tlx < intersect_brx).float()*\
    (intersect_tly < intersect_bry).float()*(
        intersect_brx - intersect_tlx)*(intersect_bry - intersect_tly)

    intersect_area = torch.relu(intersect_area)

    bound_area = (for_iou[:, :, :, 2] - for_iou[:, :, :, 0])*(for_iou[:, :, :, 3] - for_iou[:, :, :, 1])
    union = bound_area.sum(dim=0) - intersect_area

    frame_ious = intersect_area/union

    return frame_ious.mean(), frame_ious[:, -1].mean()

def calc_sensitivity(region_list):
    """ Calculates the average per-frame IoU between multiple perturbed tracks

    args:
        region_list: a list of tracks
        (each of which is a set of tlx-tly-w-h bboxes)

    Returns the mean IoU of all boxes.
    """
    # Convert the region list into batch x frames x coords to make it simpler
    num_trials = len(region_list)
    num_frames = len(region_list[0])

    tensorized = torch.zeros((num_trials, num_frames, 4))

    trial_count = 0
    for cur_trial in region_list:
        frame_count = 0
        for cur_frame in cur_trial:
            tensorized[trial_count, frame_count, :] = torch.tensor(cur_frame)
            frame_count += 1
        trial_count += 1

    # convert from w/h to brx bry.
    tensorized[:, :, 2] = tensorized[:, :, 0] + tensorized[:, :, 2]
    tensorized[:, :, 3] = tensorized[:, :, 1] + tensorized[:, :, 3]

    # We can now easily find the corners of our intersect bbox
    intersect_tlx = torch.max(tensorized[:, :, 0], dim=0)[0]
    intersect_tly = torch.max(tensorized[:, :, 1], dim=0)[0]
    intersect_brx = torch.min(tensorized[:, :, 2], dim=0)[0]
    intersect_bry = torch.min(tensorized[:, :, 3], dim=0)[0]

    intersect_area = (intersect_tlx < intersect_brx).float()*\
    (intersect_tly < intersect_bry).float()*(
        intersect_brx - intersect_tlx)*(intersect_bry - intersect_tly)

    intersect_area = torch.relu(intersect_area)

    bounds_tlx = torch.min(tensorized[:, :, 0], dim=0)[0]
    bounds_tly = torch.min(tensorized[:, :, 1], dim=0)[0]
    bounds_brx = torch.max(tensorized[:, :, 2], dim=0)[0]
    bounds_bry = torch.max(tensorized[:, :, 3], dim=0)[0]

    bounds_area = (bounds_brx - bounds_tlx)*(bounds_bry - bounds_tly)
    mean_area = ((tensorized[:, :, 2] - tensorized[:, :, 0])*\
            (tensorized[:, :, 3] - tensorized[:, :, 1])).mean(dim=0)

    return {'mean': mean_area,
            'bounds': bounds_area,
            'intersect': intersect_area}

def calc_vid_iou(regions, targets):
    """ Calculates the average IoU between a track and the ground truth.

    args:
        regions: the track bboxes (tlx tly w h)
        targets: the ground-truth bboxes (tlx tly w h)

    Returns the mean IoU of all boxes.
    """

    # Convert to a frames x coords tensor to make calculations
    num_frames = len(regions)
    tensorized = torch.zeros((2, num_frames, 4))

    for frame in range(num_frames):
        tensorized[0, frame, :] = torch.tensor(regions[frame])
        tensorized[1, frame, :] = torch.tensor(targets[frame])

    # convert from w/h to brx bry.
    tensorized[:, :, 2] = tensorized[:, :, 0] + tensorized[:, :, 2]
    tensorized[:, :, 3] = tensorized[:, :, 1] + tensorized[:, :, 3]

    # We can now easily find the corners of our intersect bbox
    intersect_tlx = torch.max(tensorized[:, :, 0], dim=0)[0]
    intersect_tly = torch.max(tensorized[:, :, 1], dim=0)[0]
    intersect_brx = torch.min(tensorized[:, :, 2], dim=0)[0]
    intersect_bry = torch.min(tensorized[:, :, 3], dim=0)[0]

    intersect_area = (intersect_tlx < intersect_brx).float()*(
        intersect_tly < intersect_bry).float()*(
            intersect_brx - intersect_tlx)*(intersect_bry - intersect_tly)

    intersect_area = torch.relu(intersect_area)

    union_area = ((tensorized[:, :, 2] - tensorized[:, :, 0])*\
        (tensorized[:, :, 3] - tensorized[:, :, 1])).sum(dim=0) - intersect_area

    vid_iou = intersect_area / union_area

    if not (vid_iou >= 0).all():
        from IPython import embed
        embed()

    return vid_iou
