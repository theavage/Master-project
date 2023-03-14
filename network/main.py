if __name__ == '__main__':
    from train import train_model
    from predict import make_predictions
    import numpy as np
    import time

    start = time.time()
    
    mode = 'test'

    if mode == 'train':
        train_model()
    
    else:

        X,radii,f_sphere, f_ball,f_stick = make_predictions()

        np.save('radii_9180_160.npy',radii)
        np.save('f_ball_9180_160.npy',f_ball)
        np.save('f_sphere_9180_160.npy',f_sphere)
        np.save('f_stick_9180_160.npy',f_stick)
        np.save('signal_9180_160.npy',X)
        
    end = time.time()
    print(end-start)
