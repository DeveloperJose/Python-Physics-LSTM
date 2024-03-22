from pathlib import Path

import cv2
import numpy as np
from numba import jit
from PIL import Image
from tqdm import tqdm

SCALE_2D = 1000
X_MAX = 4 * SCALE_2D
Y_MAX = 2.2 * SCALE_2D
STEP_2D = 0.025 * SCALE_2D
IM_ROWS = int(Y_MAX / STEP_2D)
IM_COLS = int(X_MAX / STEP_2D)
ROW_F = int(IM_ROWS / Y_MAX * SCALE_2D)
COL_F = int(IM_COLS / X_MAX * SCALE_2D)


@jit(nopython=True)
def dropout(im: np.ndarray, binomial: np.ndarray):
    # Use file ID as random number seed so the same inputs are always dropped
    orig = im.copy()
    # for i in range(3):
    for r in np.arange(1, im.shape[0]):
        for c in np.arange(1, im.shape[1]):
            # if binomial[r, c] == 0:
            im[r, c] = np.sum(im[r - 1 : r + 2, c - 1 : c + 2]) / np.sum(im[r - 1 : r + 2, c - 1 : c + 2] > 0)
    return im


def read_np_2d(filename, binomial, scaler=None):
    arr = np.load(filename).astype(np.float64)

    # input_idxs = [COLUMNS_H.index(s) for s in inputs_str]
    # output_idxs = [COLUMNS_H.index(s) for s in outputs_str]
    # X_IN = arr[:, input_idxs]
    # Y_OUT = arr[:, output_idxs]

    im = np.zeros((IM_ROWS + 1, IM_COLS + 1, 3), dtype=np.uint8)
    arr[:, 0] += 0.025
    arr[:, 0] *= SCALE_2D
    arr[:, 1] *= SCALE_2D

    for x, y, p, vu, vv in zip(arr[:, 0], arr[:, 1], arr[:, 3], arr[:, 4], arr[:, 5]):
        r = round((y * ROW_F) / SCALE_2D)
        c = round((x * COL_F) / SCALE_2D)
        im[r, c] = [vu * binomial[r, c], 0, 0]

    if scaler:
        im = scaler.transform(im)

    return im[::-1]


im_width = 160 * 8
im_height = 88 * 8
fps = 30
DROP_IN = 0.1
rng = np.random.default_rng(1738)
binomial = rng.binomial(n=1, p=1 - DROP_IN, size=(88 + 1, 160 + 1))
binomial[:, 0] = 1
binomial[0, :] = 1

for dataset in ["sled250"]:
    print("Dataset: ", dataset)
    d = Path("/home/jperez/data/") / dataset
    file_numbers = sorted([int(filepath.stem) for filepath in d.glob("*.npy")])

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    video = cv2.VideoWriter(f"{dataset}_drop01_finalv1.avi", fourcc, fps, (im_width, im_height))

    for file_id in tqdm(file_numbers):
        im = read_np_2d(d / f"{file_id}.npy", binomial)
        im[:, :, 0] = dropout(im[:, :, 0], binomial)
        video.write(cv2.resize(im, (im_width, im_height), interpolation=cv2.INTER_AREA))
    video.release()
cv2.destroyAllWindows()

# idx = 0
# while True:
#     file_id = file_numbers[idx]
#     im = cv2.resize(read_np_2d(d / f'{file_id}.npy'), (160*4, 88*4), interpolation=cv2.INTER_LANCZOS4)
#     cv2.imshow(f'Frame', im)
#     video.write(im)

#     idx += 1
#     if idx >= len(file_numbers):
#         idx = 0

#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break


# # Create a VideoCapture object and read from input file
# # If the input is the camera, pass 0 instead of the video file name
# cap = cv2.VideoCapture('video.mp4')

# # Check if camera opened successfully
# if (cap.isOpened()== False):
#   print("Error opening video stream or file")

# # Read until video is completed
# while(cap.isOpened()):
#   # Capture frame-by-frame
#   ret, frame = cap.read()
#   if ret == True:

#     # Display the resulting frame
#     cv2.imshow('Frame',frame)

#     # Press Q on keyboard to  exit
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#       break

#   # Break the loop
#   else:
#     cap = cv2.VideoCapture('video.mp4')

# print('Done')
# # When everything done, release the video capture object
# cap.release()

# # Closes all the frames
