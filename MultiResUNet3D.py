from keras.layers import Input, Conv3D, MaxPooling3D, Conv3DTranspose, concatenate, BatchNormalization, Activation, add
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.utils.vis_utils import plot_model



def conv3d_bn(x, filters, num_row, num_col, num_z, padding='same', strides=(1, 1, 1), activation='relu', name=None):
    '''
    3D Convolutional layers
    
    Arguments:
        x {keras layer} -- input layer 
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
        num_z {int} -- length along z axis in filters

    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(1, 1, 1)})
        activation {str} -- activation function (default: {'relu'})
        name {str} -- name of the layer (default: {None})
    
    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv3D(filters, (num_row, num_col, num_z), strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=4, scale=False)(x)

    if(activation==None):
        return x

    x = Activation(activation, name=name)(x)
    return x


def trans_conv3d_bn(x, filters, num_row, num_col, num_z, padding='same', strides=(2, 2, 2), name=None):
    '''
    2D Transposed Convolutional layers
    
    Arguments:
        x {keras layer} -- input layer 
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
        num_z {int} -- length along z axis in filters
    
    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(2, 2, 2)})
        name {str} -- name of the layer (default: {None})
    
    Returns:
        [keras layer] -- [output layer]
    '''

    
    x = Conv3DTranspose(filters, (num_row, num_col, num_z), strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=4, scale=False)(x)

    return x


def MultiResBlock(U, inp, alpha = 1.67):
    '''
    MultiRes Block
    
    Arguments:
        U {int} -- Number of filters in a corrsponding UNet stage
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''
    
    W = alpha * U

    shortcut = inp

    shortcut = conv3d_bn(shortcut, int(W*0.167) + int(W*0.333) + int(W*0.5), 1, 1, 1, activation=None, padding='same')

    conv3x3 = conv3d_bn(inp, int(W*0.167), 3, 3, 3, activation='relu', padding='same')

    conv5x5 = conv3d_bn(conv3x3, int(W*0.333), 3, 3, 3, activation='relu', padding='same')

    conv7x7 = conv3d_bn(conv5x5, int(W*0.5), 3, 3, 3, activation='relu', padding='same')

    out = concatenate([conv3x3, conv5x5, conv7x7], axis=4)
    out = BatchNormalization(axis=4)(out)
    
    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=4)(out)

    return out

def ResPath(filters, length, inp):
    '''
    ResPath
    
    Arguments:
        filters {int} -- [description]
        length {int} -- length of ResPath
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''

    shortcut = inp
    shortcut = conv3d_bn(shortcut, filters , 1, 1, 1, activation=None, padding='same')

    out = conv3d_bn(inp, filters, 3, 3, 3, activation='relu', padding='same')

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=4)(out)

    for i in range(length-1):

        shortcut = out
        shortcut = conv3d_bn(shortcut, filters , 1, 1, 1, activation=None, padding='same')

        out = conv3d_bn(out, filters, 3, 3, 3, activation='relu', padding='same')        

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization(axis=4)(out)


    return out


def MultiResUnet3D(height, width, z, n_channels):
    '''
    MultiResUNet3D
    
    Arguments:
        height {int} -- height of image 
        width {int} -- width of image
        z {int} -- length along z axis 
        n_channels {int} -- number of channels in image
    
    Returns:
        [keras model] -- MultiResUNet3D model
    '''


    inputs = Input((height, width, z, n_channels))

    mresblock1 = MultiResBlock(32, inputs) 
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(mresblock1)
    mresblock1 = ResPath(32, 4, mresblock1) 

    mresblock2 = MultiResBlock(32*2, pool1)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(mresblock2)
    mresblock2 = ResPath(32*2, 3,mresblock2) 

    mresblock3 = MultiResBlock(32*4, pool2)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(mresblock3)
    mresblock3 = ResPath(32*4, 2,mresblock3) 

    mresblock4 = MultiResBlock(32*8, pool3)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(mresblock4)
    mresblock4 = ResPath(32*8, 1,mresblock4) 

    mresblock5 = MultiResBlock(32*16, pool4)

    up6 = concatenate([Conv3DTranspose(32*8, (2, 2, 2), strides=(2, 2, 2), padding='same')(mresblock5), mresblock4], axis=4)
    mresblock6 = MultiResBlock(32*8,up6)
    
    up7 = concatenate([Conv3DTranspose(32*4, (2, 2, 2), strides=(2, 2, 2), padding='same')(mresblock6), mresblock3], axis=4)
    mresblock7 = MultiResBlock(32*4,up7)

    up8 = concatenate([Conv3DTranspose(32*2, (2, 2, 2), strides=(2, 2, 2), padding='same')(mresblock7), mresblock2], axis=4)
    mresblock8 = MultiResBlock(32*2,up8)

    up9 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(mresblock8), mresblock1], axis=4)
    mresblock9 = MultiResBlock(32,up9)

    conv10 = conv3d_bn(mresblock9 , 1, 1, 1, 1, activation='sigmoid')

    model = Model(inputs=[inputs], outputs=[conv10])

    return model


def main():

    # Define the model

    model = MultiResUnet3D(80, 80, 48, 4)
    print(model.summary())

    


if __name__ == '__main__':
    main()