import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from skimage import morphology, io, color, exposure, img_as_float, transform
from matplotlib import pyplot as plt

def loadDataGeneral(df, path, im_shape):
    X, y = [], []
    for i, item in df.iterrows():
        img = img_as_float(io.imread(path + item[0]))
        img = transform.resize(img, im_shape)
        img = exposure.equalize_hist(img)
        img = np.expand_dims(img, -1)
        X.append(img)
    X = np.array(X)
    X -= X.mean()
    X /= X.std()

    print('### Dataset loaded')
    print('\t{}'.format(path))
    print('\t{}\t{}'.format(X.shape, 0))
    print('\tX:{:.1f}-{:.1f}\ty:{:.1f}-{:.1f}\n'.format(X.min(), X.max(), 0, 0))
    print('\tX.mean = {}, X.std = {}'.format(X.mean(), X.std()))
    return X

def remove_small_regions(img, size):
    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img

if __name__ == '__main__':
    # Path to csv-file. File should contain X-ray filenames as first column,
    # mask filenames as second column.
    csv_path = 'covidnames.csv'
    # Path to the folder with images. Images will be read from path + path_from_csv
    path = 'Covid_Xrays/'

    df = pd.read_csv(csv_path)

    # Load test data
    im_shape = (256, 256)
    X = loadDataGeneral(df, path, im_shape)

    n_test = X.shape[0]
    inp_shape = X[0].shape

    # Load model
    model_name = 'lung-segmentation-2d/trained_model.hdf5'
    UNet = load_model(model_name)

    # For inference standard keras ImageGenerator can be used.
    test_gen = ImageDataGenerator(rescale=1.)

    prs = []
    i = 0
    plt.figure(figsize=(10, 10))
    for xx in test_gen.flow(X, batch_size=1):
        img = exposure.rescale_intensity(np.squeeze(xx), out_range=(0,1))
        pred = UNet.predict(xx)[..., 0].reshape(inp_shape[:2])

        pr = pred > 0.5

        pr = remove_small_regions(pr, 0.02 * np.prod(im_shape))

        prs.append(pr)

        plt.subplot(2, 2, 1)
        plt.title('Processed ' + df.iloc[i][0])
        plt.axis('off')
        plt.imshow(img, cmap='gray')

        plt.subplot(2, 2, 2)
        plt.title('Prediction')
        plt.axis('off')
        plt.imshow(pred, cmap='jet')
        plt.tight_layout()
        plt.savefig('Covid_Xrays/prediction{}.png'.format(i))

        i += 1
        if i == n_test:
            break
