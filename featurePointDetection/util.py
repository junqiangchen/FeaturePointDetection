import pandas as pd
import numpy as np
import cv2


def show_result(x, y):
    img = x.reshape(96, 96) * 255.
    img = np.clip(img, 0, 255).astype('uint8')
    for step in range(0, 30, 2):
        cv2.circle(img, center=(int(np.round(y[step] * 96)), int(np.round(y[step + 1] * 96))), radius=1,
                   color=(255, 0, 0))
    img = cv2.resize(img, dsize=(512, 512))
    cv2.imwrite('img.jpg', img)


def load_image_label(path, test=False):
    df = pd.read_csv(path)
    # The Image column has pixel values separated by space
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    print(df.count())
    df = df.dropna()  # drop all rows that have missing values in them
    X = np.vstack(df['Image'].values) / 255.
    X = X.astype(np.float32)
    if not test:
        Y = df.iloc[:, 0:-1].values
        Y = (Y - 0) / 96.
        Y = Y.astype(np.float32)
    else:
        Y = None
    return X, Y
