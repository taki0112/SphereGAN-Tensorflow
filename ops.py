import tensorflow as tf

##################################################################################
# Initialization
##################################################################################

# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# Truncated_normal : tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
# Orthogonal : tf.orthogonal_initializer(1.0) / relu = sqrt(2), the others = 1.0

##################################################################################
# Regularization
##################################################################################

# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)

weight_init = tf.contrib.layers.xavier_initializer()
weight_regularizer = None
weight_regularizer_fully = None

##################################################################################
# Layer
##################################################################################

# pad = ceil[ (kernel - stride) / 2 ]

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
    with tf.variable_scope(scope):
        if pad > 0:
            h = x.get_shape().as_list()[1]
            if h % stride == 0:
                pad = pad * 2
            else:
                pad = max(kernel - (h % stride), 0)

            pad_top = pad // 2
            pad_bottom = pad - pad_top
            pad_left = pad // 2
            pad_right = pad - pad_left

            if pad_type == 'zero':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            if pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias)

        return x


def deconv(x, channels, kernel=4, stride=2, padding='SAME', use_bias=True, sn=False, scope='deconv_0'):
    with tf.variable_scope(scope):
        x_shape = x.get_shape().as_list()

        if padding == 'SAME':
            output_shape = [x_shape[0], x_shape[1] * stride, x_shape[2] * stride, channels]

        else:
            output_shape = [x_shape[0], x_shape[1] * stride + max(kernel - stride, 0),
                            x_shape[2] * stride + max(kernel - stride, 0), channels]

        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, channels, x.get_shape()[-1]], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d_transpose(x, filter=spectral_norm(w), output_shape=output_shape,
                                       strides=[1, stride, stride, 1], padding=padding)

            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                           kernel_size=kernel, kernel_initializer=weight_init,
                                           kernel_regularizer=weight_regularizer,
                                           strides=stride, padding=padding, use_bias=use_bias)

        return x

def fully_connected(x, units, use_bias=True, sn=False, scope='linear'):
    with tf.variable_scope(scope):
        x = flatten(x)
        shape = x.get_shape().as_list()
        channels = shape[-1]

        if sn:
            w = tf.get_variable("kernel", [channels, units], tf.float32,
                                initializer=weight_init, regularizer=weight_regularizer_fully)
            if use_bias:
                bias = tf.get_variable("bias", [units],
                                       initializer=tf.constant_initializer(0.0))

                x = tf.matmul(x, spectral_norm(w)) + bias
            else:
                x = tf.matmul(x, spectral_norm(w))

        else:
            x = tf.layers.dense(x, units=units, kernel_initializer=weight_init,
                                kernel_regularizer=weight_regularizer_fully,
                                use_bias=use_bias)

        return x


def flatten(x) :
    return tf.layers.flatten(x)


##################################################################################
# Residual-block
##################################################################################

def resblock(x_init, channels, use_bias=True, is_training=True, sn=False, scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('init') :
            x = layer_norm(x_init)
            x = lrelu(x, 0.2)

        with tf.variable_scope('res1'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)
            x = layer_norm(x)
            x = lrelu(x, 0.2)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)

        return x + x_init


def resblock_up(x_init, channels, use_bias=True, is_training=True, sn=False, scope='resblock_up'):
    with tf.variable_scope(scope):
        with tf.variable_scope('init') :
            x = batch_norm(x_init, is_training)
            x = relu(x)

        with tf.variable_scope('res1'):
            x = upsample(x)
            x = conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)
            x = batch_norm(x, is_training)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)


        with tf.variable_scope('shortcut'):
            x_init = upsample(x_init)
            x_init = conv(x_init, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)

    return x + x_init

def resblock_down(x_init, channels, use_bias=True, is_training=True, sn=False, scope='resblock_down'):
    with tf.variable_scope(scope):
        with tf.variable_scope('init') :
            x = layer_norm(x_init)
            x = lrelu(x, 0.2)

        with tf.variable_scope('res1'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)
            x = layer_norm(x)
            x = lrelu(x, 0.2)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)
            x = downsample(x)

        with tf.variable_scope('shortcut'):
            x_init = conv(x_init, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)
            x_init = downsample(x_init)

    return x + x_init

def resblock_2(x_init, channels, use_bias=True, is_training=True, sn=False, scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)
            x = layer_norm(x)
            x = lrelu(x, 0.2)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)
            x = layer_norm(x)

        return x + x_init


def resblock_up_2(x_init, channels, use_bias=True, is_training=True, sn=False, scope='resblock_up'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = deconv(x_init, channels, kernel=3, stride=2, use_bias=use_bias, sn=sn)
            x = batch_norm(x, is_training)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = deconv(x, channels, kernel=3, stride=1, use_bias=use_bias, sn=sn)
            x = batch_norm(x, is_training)

        with tf.variable_scope('skip'):
            x_init = deconv(x_init, channels, kernel=3, stride=2, use_bias=use_bias, sn=sn)

    return relu(x + x_init)

def resblock_down_2(x_init, channels, use_bias=True, is_training=True, sn=False, scope='resblock_down'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=2, pad=1, use_bias=use_bias, sn=sn)
            x = layer_norm(x)
            x = lrelu(x, 0.2)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)
            x = layer_norm(x)

        with tf.variable_scope('skip'):
            x_init = conv(x_init, channels, kernel=3, stride=2, pad=1, use_bias=use_bias, sn=sn)

    return lrelu(x + x_init, 0.2)


##################################################################################
# Sampling
##################################################################################

def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2])

    return gap

def downsample(x):
    x = tf.add_n([x[:, ::2, ::2, :], x[:, 1::2, ::2, :], x[:, ::2, 1::2, :], x[:, 1::2, 1::2, :]]) / 4.0

    return x

def upsample(x):
    x = tf.concat([x, x, x, x], axis=-1)
    x = tf.depth_to_space(x, 2)

    return x

##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)

##################################################################################
# Normalization function
##################################################################################

def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf.layers.batch_normalization(x,
                                         momentum=0.9,
                                         epsilon=1e-05,
                                         training=is_training,
                                         name=scope)

def layer_norm(x, scope='layer_norm'):
    return tf.contrib.layers.layer_norm(x,
                                        center=True, scale=True,
                                        scope=scope)

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """

        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

##################################################################################
# Loss function
##################################################################################

def discriminator_loss(loss_func, real, fake, moment=3):
    real_loss = 0
    fake_loss = 0
    bs, c = real.get_shape().as_list()

    if loss_func.__contains__('wgan') :
        real_loss = -tf.reduce_mean(real)
        fake_loss = tf.reduce_mean(fake)

    if loss_func == 'lsgan' :
        real_loss = tf.reduce_mean(tf.squared_difference(real, 1.0))
        fake_loss = tf.reduce_mean(tf.square(fake))

    if loss_func == 'gan' or loss_func == 'dragan' :
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))

    if loss_func == 'hinge' :
        real_loss = tf.reduce_mean(relu(1.0 - real))
        fake_loss = tf.reduce_mean(relu(1.0 + fake))

    if loss_func == 'sphere' :
        north_pole = tf.one_hot(tf.tile([c], multiples=[bs]), depth=c+1) # [bs, c+1] -> [0, 0, 0, ... , 1]

        real_projection = inverse_stereographic_projection(real)
        fake_projection = inverse_stereographic_projection(fake)

        for i in range(1, moment+1) :
            real_loss += -tf.reduce_mean(tf.pow(sphere_loss(real_projection, north_pole), i))
            fake_loss += tf.reduce_mean(tf.pow(sphere_loss(fake_projection, north_pole), i))

    loss = real_loss + fake_loss

    return loss

def generator_loss(loss_func, fake, moment=3):
    fake_loss = 0
    bs, c = fake.get_shape().as_list()

    if loss_func.__contains__('wgan') :
        fake_loss = -tf.reduce_mean(fake)

    if loss_func == 'lsgan' :
        fake_loss = tf.reduce_mean(tf.squared_difference(fake, 1.0))

    if loss_func == 'gan' or loss_func == 'dragan' :
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))

    if loss_func == 'hinge' :
        fake_loss = -tf.reduce_mean(fake)

    if loss_func == 'sphere' :
        north_pole = tf.one_hot(tf.tile([c], multiples=[bs]), depth=c+1)  # [bs, c+1] -> [0, 0, 0, ... , 1]


        fake_projection = inverse_stereographic_projection(fake)

        for i in range(1, moment+1) :
            fake_loss += -tf.reduce_mean(tf.pow(sphere_loss(fake_projection, north_pole), i))

    loss = fake_loss

    return loss

def inverse_stereographic_projection(x) :

    x_u = tf.transpose(2 * x) / (tf.pow(tf.norm(x, axis=-1), 2) + 1.0)
    x_v = (tf.pow(tf.norm(x_u, axis=0, keepdims=True), 2) - 1.0) / (tf.pow(tf.norm(x_u, axis=0, keepdims=True), 2) + 1.0)

    x_projection = tf.transpose(tf.concat([x_u, x_v], axis=0))

    return x_projection

def sphere_loss(x, y) :

    # x_2 = tf.pow(tf.norm(x, axis=-1, keepdims=True), 2)
    # y_2 = tf.pow(tf.norm(y, axis=-1, keepdims=True), 2)
    #
    # denominator = x_2 * y_2 - x_2 - y_2 + 4*x*y + 1.0
    # numerator = (x_2 + 1.0) * (y_2 + 1.0)
    #
    # loss = tf.math.acos(denominator / numerator)

    loss = tf.math.acos(tf.keras.backend.batch_dot(x, y, axes=1))

    return loss
