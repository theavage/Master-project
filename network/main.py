if __name__ == '__main__':
    from train import train_model
    from predict import make_predictions
    import numpy as np

    mode = 'test'

    if mode == 'train':
        train_model()

    else:
        X,radii,f_sphere, f_ball,f_stick = make_predictions()

        np.save('radii_test.npy',radii)
        np.save('f_ball_test.npy',f_ball)
        np.save('f_sphere_test.npy',f_sphere)
        np.save('f_stick_test.npy',f_stick)
        np.save('signal_test.npy',X)