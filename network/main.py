if __name__ == '__main__':
    from train import train_model
    from predict import make_predictions
    import numpy as np

    mode = 'train'

    if mode == 'train':
        train_model()

    else:
        X,radii,f_sphere, f_ball,f_stick = make_predictions()
        
        np.save('signal_MRI_10.npy',X)
        np.save('radii_MRI_10.npy',radii)
        np.save('f_ball_MRI_10.npy',f_ball)
        np.save('f_sphere_MRI_10.npy',f_sphere)
        np.save('f_stick_MRI_10.npy',f_stick)