# import the necessary packages
import numpy as np
import cv2
import imutils


def color_transfer(source, target, clip=True, preserve_paper=True):
    """
    Transfers the color distribution from the source to the target
    image using the mean and standard deviations of the L*a*b*
    color space.

    This implementation is (loosely) based on to the "Color Transfer
    between Images" paper by Reinhard et al., 2001.

    Parameters:
    -------
    source: NumPy array
    OpenCV image in BGR color space (the source image)
    target: NumPy array
    OpenCV image in BGR color space (the target image)
    clip: Should components of L*a*b* image be scaled by np.clip before
        converting back to BGR color space?
        If False then components will be min-max scaled appropriately.
        Clipping will keep target image brightness truer to the input.
        Scaling will adjust image brightness to avoid washed out portions
        in the resulting color transfer that can be caused by clipping.
    preserve_paper: Should color transfer strictly follow methodology
        laid out in original paper? The method does not always produce
        aesthetically pleasing results.
        If False then L*a*b* components will scaled using the reciprocal of
        the scaling factor proposed in the paper.  This method seems to produce
        more consistently aesthetically pleasing results

    Returns:
    -------
    transfer: NumPy array
        OpenCV image (w, h, 3) NumPy array (uint8)
    """
    # convert the images from the RGB to L*ab* color space, being
    # sure to utilizing the floating point data type (note: OpenCV
    # expects floats to be 32-bit, so use that instead of 64-bit)
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    # compute color statistics for the source and target images
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)

    # subtract the means from the target image
    (l, a, b) = cv2.split(target)
    l -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar

    if preserve_paper:
        # scale by the standard deviations using paper proposed factor
        l = (lStdTar / lStdSrc) * l
        a = (aStdTar / aStdSrc) * a
        b = (bStdTar / bStdSrc) * b
    else:
        # scale by the standard deviations using reciprocal of paper proposed factor
        l = (lStdSrc / lStdTar) * l
        a = (aStdSrc / aStdTar) * a
        b = (bStdSrc / bStdTar) * b

    # add in the source mean
    l += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc

    # clip/scale the pixel intensities to [0, 255] if they fall
    # outside this range
    l = _scale_array(l, clip=clip)
    a = _scale_array(a, clip=clip)
    b = _scale_array(b, clip=clip)

    # merge the channels together and convert back to the RGB color
    # space, being sure to utilize the 8-bit unsigned integer data
    # type
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)

    # return the color transferred image
    return transfer


def auto_color_transfer(source, target):
    """Pick color_transfer result truest to source image color

    Applies color_transfer with all possible combinations of the clip & preserve_paper arguments.
    Mean absolute error (MAE) is computed for the HSV channels of each result and the source image.
    The best_result that minimizes the MAE is returned as well as a montage of all candidate results.

    Parameters:
    -------
    source: NumPy array
    OpenCV image in BGR color space (the source image)
    target: NumPy array
    OpenCV image in BGR color space (the target image)

    Returns:
    -------
    tuple: (best_result, comparison)
        best_result: NumPy array
        result that minimizes mean absolute error between compared to source image in HSV color space
        comparison:  NumPy array
        image showing the results of all combinations of color_transfer options
    """
    # get mean L*a*b* stats from source image for comparison
    hsv_source = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    hsv_means_src = np.array(image_stats(hsv_source)[::2])

    # iterate through all 4 options for toggling color transfer
    bools = [True, False]
    candidates = []
    best_result = None
    best_abs_err = float('inf')
    for clip in bools:
        for preserve_paper in bools:
            # create candidate image from options of this iteration
            candidate = color_transfer(source, target, clip, preserve_paper)
            # get mean L*a*b* stats from candidate image for comparison
            hsv_candidate = cv2.cvtColor(candidate, cv2.COLOR_BGR2HSV)
            hsv_means_cand = np.array(image_stats(hsv_candidate)[::2])

            # calc mean absolute error across L*a*b* means
            mean_abs_err = np.mean(np.abs(hsv_means_src - hsv_means_cand))

            # propose new truest result if found new smallest mae
            if mean_abs_err < best_abs_err:
                best_result = candidate[:]

            candidates.append(candidate)

    comparison = np.hstack((np.vstack(candidates[:2]),
                            np.vstack(candidates[2:])))
    comparison = _bool_matrix_border(comparison)

    return best_result, comparison


def _bool_matrix_border(comparison_image):
    """Apply table formatting for comparison of color_transfer options

    Parameters:
    -------
    target: NumPy array
    OpenCV image in BGR color space (the comparison image produced in auto_color_transfer)

    Returns:
    -------
    comparison: NumPy array
    OpenCV image in BGR color space with borders applied to easily compare the different
    results of the auto_color_transfer
    """
    # 200 seems to work well as border size
    border_size = 200

    # put black border on top and left of input image
    h, w = comparison_image.shape[:2]
    top = np.zeros(w * border_size, dtype='uint8').reshape(border_size, w)
    left = np.zeros((h + border_size) * border_size, dtype='uint8').reshape(h + border_size, border_size)
    top = cv2.cvtColor(top, cv2.COLOR_GRAY2BGR)
    left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    bordered_comparison_image = np.vstack((top, comparison_image))
    bordered_comparison_image = np.hstack((left, bordered_comparison_image))

    # add text for clip arg options to top border
    top_title_loc = (border_size, 75)
    top_true_loc = (border_size, 190)
    top_false_loc = (int(border_size + w / 2), 190)
    cv2.putText(bordered_comparison_image, 'Clip', top_title_loc,
                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
    cv2.putText(bordered_comparison_image, 'True', top_true_loc,
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    cv2.putText(bordered_comparison_image, 'False', top_false_loc,
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

    # rotate 90 degrees for writing text to left border
    bordered_comparison_image = imutils.rotate_bound(bordered_comparison_image, 90)

    # add text for preserve paper arg options to left border
    top_title_loc = (5, 75)
    top_true_loc = (5 + int(h / 2), 190)
    top_false_loc = (5, 190)
    cv2.putText(bordered_comparison_image, 'Preserve Paper', top_title_loc,
                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
    cv2.putText(bordered_comparison_image, 'True', top_true_loc,
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    cv2.putText(bordered_comparison_image, 'False', top_false_loc,
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

    # rotate -90 degrees to return image in correct orientation
    bordered_comparison_image = imutils.rotate_bound(bordered_comparison_image, -90)

    return bordered_comparison_image


def image_stats(image):
    """
    Parameters:
    -------
    image: NumPy array
        OpenCV image in L*a*b* color space

    Returns:
    -------
    Tuple of mean and standard deviations for the L*, a*, and b*
    channels, respectively
    """
    # compute the mean and standard deviation of each channel
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())

    # return the color statistics
    return lMean, lStd, aMean, aStd, bMean, bStd


def _min_max_scale(arr, new_range=(0, 255)):
    """
    Perform min-max scaling to a NumPy array

    Parameters:
    -------
    arr: NumPy array to be scaled to [new_min, new_max] range
    new_range: tuple of form (min, max) specifying range of
        transformed array

    Returns:
    -------
    NumPy array that has been scaled to be in
    [new_range[0], new_range[1]] range
    """
    # get array's current min and max
    mn = arr.min()
    mx = arr.max()

    # check if scaling needs to be done to be in new_range
    if mn < new_range[0] or mx > new_range[1]:
        # perform min-max scaling
        scaled = (new_range[1] - new_range[0]) * (arr - mn) / (mx - mn) + new_range[0]
    else:
        # return array if already in range
        scaled = arr

    return scaled


def _scale_array(arr, clip=True):
    """
    Trim NumPy array values to be in [0, 255] range with option of
    clipping or scaling.

    Parameters:
    -------
    arr: array to be trimmed to [0, 255] range
    clip: should array be scaled by np.clip? if False then input
        array will be min-max scaled to range
        [max([arr.min(), 0]), min([arr.max(), 255])]

    Returns:
    -------
    NumPy array that has been scaled to be in [0, 255] range
    """
    if clip:
        scaled = np.clip(arr, 0, 255)
    else:
        scale_range = (max([arr.min(), 0]), min([arr.max(), 255]))
        scaled = _min_max_scale(arr, new_range=scale_range)

    return scaled
