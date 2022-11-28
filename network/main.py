if __name__ == '__main__':
    from train import train_model
    from predict import make_predictions
    import numpy as np

    mode = 'pred'

    if mode == 'train':
        train_model()

    else:
        pred = make_predictions()
        np.save('signal.npy',pred)