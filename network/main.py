if __name__ == '__main__':
    from train import train_model
    from predict import make_predictions
    import numpy as np

    mode = 'train'

    if mode == 'train':
        train_model()

    else:
        X,radii,f_sphere, f_ball,f_stick = make_predictions()

        np.save('radii_3466_4_fast.npy',radii)
        np.save('f_ball_3466_4_fast.npy',f_ball)
        np.save('f_sphere_3466_4_fast.npy',f_sphere)
        np.save('f_stick_3466_4_fast.npy',f_stick)