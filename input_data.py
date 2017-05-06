import scipy.io as sio
#import h5py

data = sio.loadmat('./data/test_64_32_source1999_ellipse_value_6.mat')
#data = h5py.File('./data/test_64_circle_1e4.mat')
Sig_set_reshape = data["Sig_set_reshape"]
N_test = data["N_test"]
U_ob_reshape = data["U_ob_reshape"]
def input_data(test):
    if test:
        x = Sig_set_reshape[int(N_test[0][0] * 0.8):]
        x = x.reshape(-1,66,66,1)
        y = U_ob_reshape[int(N_test[0][0] * 0.8):]
        y = y.reshape(-1,32,32,1)
    else:
        x = Sig_set_reshape[:int(N_test[0][0] * 0.8)]
        x = x.reshape(-1,66,66,1)
        y = U_ob_reshape[:int(N_test[0][0] * 0.8)]
        y = y.reshape(-1,32,32,1)
    return x,y
