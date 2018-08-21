from keras.losses import binary_crossentropy
import keras.backend as K


def dice_metric(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coeff_hard(y_true, y_pred): 
    smooth = 1. 
    y_true_f = K.flatten(y_true) 
    y_pred_f = K.round(K.flatten(y_pred)) 
    intersection = K.sum(y_true_f * y_pred_f) 
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth) 
    return score




def dice_metric_true(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    return y_true_f

def dice_metric_pred(y_true, y_pred):
    y_pred_f = K.flatten(y_pred)
    return y_pred_f

def dice_metric_intersection(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def focus_loss(y_true,y_pred):
    gamma = .01
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    loss = y_true*K.log(y_pred+K.epsilon())*(1-y_pred+K.epsilon())**gamma + (1-y_true)*K.log(1-y_pred+K.epsilon())*(y_pred+K.epsilon())**gamma
    return (-K.sum(loss))



def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def jacard_focus_loss_coef(y_true, y_pred):
    return jacard_coef_loss(y_true, y_pred)+ focus_loss(y_true,y_pred)



def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)



def dice_jacard_loss(y_true, y_pred):
    return jacard_coef_loss(y_true, y_pred) + (1 - dice_metric(y_true, y_pred))


def dice_jacard_binary_loss(y_true, y_pred):
    return jacard_coef_loss(y_true, y_pred) + (1 - dice_metric(y_true, y_pred))*.2 + binary_crossentropy(y_true, y_pred)*.05


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) +  jacard_coef_loss(y_true, y_pred)

def weighted_dice_loss_orig(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    averaged_mask = K.pool2d(
        y_true, pool_size=(41, 41), strides=(1, 1), padding='same', pool_mode='avg')
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = 1 - weighted_dice_coeff_orig(y_true, y_pred, weight)
    return loss

def weighted_jacard_loss_orig(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    averaged_mask = K.pool2d(
        y_true, pool_size=(41, 41), strides=(1, 1), padding='same', pool_mode='avg')
    border = K.cast(K.greater(averaged_mask, 0.05), 'float32') * K.cast(K.less(averaged_mask, 0.95), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border *2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_jaccard_coeff_orig(y_true, y_pred, weight)
    return loss


def weighted_jaccard_coeff_orig(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = (m1 * m2)
    score = -(K.sum(w * intersection) + 1.) / (K.sum(w * m1) + K.sum(w * m2) - K.sum(w * intersection)+1 )
    #(intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)
    return score

def weighted_dice_coeff_orig(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    return score


def weighted_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    if K.int_shape(y_pred)[0] == 128:
        kernel_size = 11
    elif K.int_shape(y_pred[0]) == 1280:
        kernel_size = 41
    elif K.int_shape(y_pred[0]) == 512:
        kernel_size = 21
    elif K.int_shape(y_pred[0]) == 1024:
        kernel_size = 41
    else:
        raise ValueError('Unexpected image size')
    averaged_mask = K.pool2d(
        y_true, pool_size=(kernel_size, kernel_size), strides=(1, 1), padding='same', pool_mode='avg')
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = 1 - weighted_dice_coeff(y_true, y_pred, weight)
    return loss


def weighted_bce_loss(y_true, y_pred, weight):
    # avoiding overflow
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))

    # https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
    loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * \
                                          (K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)


def weighted_bce_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    if K.int_shape(y_pred)[0] == 128:
        kernel_size = 11
    elif K.int_shape(y_pred[0]) == 256:
        kernel_size = 21
    elif K.int_shape(y_pred[0]) == 512:
        kernel_size = 21
    elif K.int_shape(y_pred[0]) == 1024:
        kernel_size = 41
    else:
        raise ValueError('Unexpected image size')
    averaged_mask = K.pool2d(
        y_true, pool_size=(kernel_size, kernel_size), strides=(1, 1), padding='same', pool_mode='avg')
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_bce_loss(y_true, y_pred, weight) + (1 - weighted_dice_coeff(y_true, y_pred, weight))
    return loss
