if __name__ == '__main__':
    from train import train_model
    from predict import make_predictions
    import numpy as np

    mode = 'pred'

    if mode == 'train':
        train_model()

    else:
        X,radii,f_sphere, f_ball,f_stick = make_predictions()
        
        np.save('signal_50.npy',X)
        np.save('radii_50.npy',radii)
        np.save('f_ball_50.npy',f_ball)
        np.save('f_sphere_50.npy',f_sphere)
        np.save('f_stick_50.npy',f_stick)