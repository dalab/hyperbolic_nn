import util
import tensorflow as tf

class EuclRNN(tf.nn.rnn_cell.RNNCell):

    def __init__(self, num_units, dtype):
        self._num_units = num_units
        self.built = False
        self.__dtype = dtype
        self.eucl_vars = []

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            if not self.built:
                inputs_shape = inputs.get_shape()
                print('Init RNN cell')
                if inputs_shape[1].value is None:
                    raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                                     % inputs_shape)
                input_depth = inputs_shape[1].value

                self.W = tf.get_variable(
                    'W', dtype= self.__dtype,
                    shape=[self._num_units, self._num_units],
                    initializer=tf.contrib.layers.xavier_initializer())
                self.eucl_vars.append(self.W)

                self.U = tf.get_variable(
                    'U', dtype= self.__dtype,
                    shape=[input_depth, self._num_units],
                    initializer=tf.contrib.layers.xavier_initializer())
                self.eucl_vars.append(self.U)

                self.b = tf.get_variable(
                    'b', dtype= self.__dtype,
                    shape=[1, self._num_units],
                    initializer=tf.constant_initializer(0.0))
                self.eucl_vars.append(self.b)

                self.built = True

            new_h = tf.tanh(tf.matmul(state, self.W) + tf.matmul(inputs, self.U) + self.b)

        return new_h, new_h

################################################################################################

class EuclGRU(tf.nn.rnn_cell.RNNCell):
    """
    From https://en.wikipedia.org/wiki/Gated_recurrent_unit and
    http://colah.github.io/posts/2015-08-Understanding-LSTMs/
    """

    def __init__(self, num_units, dtype):
        self._num_units = num_units
        self.__dtype = dtype
        self.built = False

        self.eucl_vars = []

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            if not self.built:
                inputs_shape = inputs.get_shape()
                print('Init GRU cell')
                if inputs_shape[1].value is None:
                    raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                                     % inputs_shape)
                input_depth = inputs_shape[1].value

                self.Wz = tf.get_variable('W_z', dtype= self.__dtype,
                                          shape=[self._num_units, self._num_units],
                                          initializer=tf.contrib.layers.xavier_initializer())
                self.eucl_vars.append(self.Wz)
                self.Uz = tf.get_variable('U_z', dtype= self.__dtype,
                                          shape=[input_depth, self._num_units],
                                          initializer=tf.contrib.layers.xavier_initializer())
                self.eucl_vars.append(self.Uz)
                self.bz = tf.get_variable('b_z', dtype= self.__dtype,
                                          shape=[1, self._num_units],
                                          initializer=tf.constant_initializer(0.0))
                self.eucl_vars.append(self.bz)

                self.Wr = tf.get_variable('W_r', dtype= self.__dtype,
                                          shape=[self._num_units, self._num_units],
                                          initializer=tf.contrib.layers.xavier_initializer())
                self.eucl_vars.append(self.Wr)
                self.Ur = tf.get_variable('U_r', dtype= self.__dtype,
                                          shape=[input_depth, self._num_units],
                                          initializer=tf.contrib.layers.xavier_initializer())
                self.eucl_vars.append(self.Ur)
                self.br = tf.get_variable('b_r', dtype= self.__dtype,
                                          shape=[1, self._num_units],
                                          initializer=tf.constant_initializer(0.0))
                self.eucl_vars.append(self.br)

                self.Wh = tf.get_variable('W_h', dtype= self.__dtype,
                                          shape=[self._num_units, self._num_units],
                                          initializer=tf.contrib.layers.xavier_initializer())
                self.eucl_vars.append(self.Wh)
                self.Uh = tf.get_variable('U_h', dtype= self.__dtype,
                                          shape=[input_depth, self._num_units],
                                          initializer=tf.contrib.layers.xavier_initializer())
                self.eucl_vars.append(self.Uh)
                self.bh = tf.get_variable('b_h', dtype= self.__dtype,
                                          shape=[1, self._num_units],
                                          initializer=tf.constant_initializer(0.0))
                self.eucl_vars.append(self.bh)

                self.built = True

            z = tf.nn.sigmoid(tf.matmul(state, self.Wz) + tf.matmul(inputs, self.Uz) + self.bz)
            r = tf.nn.sigmoid(tf.matmul(state, self.Wr) + tf.matmul(inputs, self.Ur) + self.br)
            h_tilde = tf.tanh(tf.matmul(r * state, self.Wh) + tf.matmul(inputs, self.Uh) + self.bh)

            new_h = (1 - z) * state + z * h_tilde
        return new_h, new_h

################################################################################################


class HypRNN(tf.nn.rnn_cell.RNNCell):

    def __init__(self,
                 num_units,
                 inputs_geom,
                 bias_geom,
                 c_val,
                 non_lin,
                 fix_biases,
                 fix_matrices,
                 matrices_init_eye,
                 dtype):
        self._num_units = num_units
        self.c_val = c_val
        self.built = False
        self.__dtype = dtype
        self.non_lin = non_lin
        assert self.non_lin in ['id', 'relu', 'tanh', 'sigmoid']

        self.bias_geom = bias_geom
        self.inputs_geom = inputs_geom
        assert self.inputs_geom in ['eucl', 'hyp']
        assert self.bias_geom in ['eucl', 'hyp']

        self.fix_biases = fix_biases
        self.fix_matrices = fix_matrices
        if matrices_init_eye or self.fix_matrices:
            self.matrix_initializer = tf.initializers.identity()
        else:
            self.matrix_initializer = tf.contrib.layers.xavier_initializer()

        self.eucl_vars = []
        self.hyp_vars = []

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    # Performs the hyperbolic version of the operation Wh + Ux + b.
    def one_rnn_transform(self, W, h, U, x, b):
        hyp_x = x
        if self.inputs_geom == 'eucl':
            hyp_x = util.tf_exp_map_zero(x, self.c_val)

        hyp_b = b
        if self.bias_geom == 'eucl':
            hyp_b = util.tf_exp_map_zero(b, self.c_val)

        W_otimes_h = util.tf_mob_mat_mul(W, h, self.c_val)
        U_otimes_x = util.tf_mob_mat_mul(U, hyp_x, self.c_val)
        Wh_plus_Ux = util.tf_mob_add(W_otimes_h, U_otimes_x, self.c_val)
        result = util.tf_mob_add(Wh_plus_Ux, hyp_b, self.c_val)
        return result


    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            if not self.built:
                inputs_shape = inputs.get_shape()
                print('Init RNN cell')
                if inputs_shape[1].value is None:
                    raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                                     % inputs_shape)
                input_depth = inputs_shape[1].value

                self.W = tf.get_variable(
                    'W', dtype= self.__dtype,
                    shape=[self._num_units, self._num_units],
                    trainable=(not self.fix_matrices),
                    initializer=self.matrix_initializer)
                if not self.fix_matrices:
                    self.eucl_vars.append(self.W)

                self.U = tf.get_variable(
                    'U', dtype= self.__dtype,
                    shape=[input_depth, self._num_units],
                    trainable=(not self.fix_matrices),
                    initializer=self.matrix_initializer)
                if not self.fix_matrices:
                    self.eucl_vars.append(self.U)

                self.b = tf.get_variable(
                    'b', dtype= self.__dtype,
                    shape=[1, self._num_units],
                    trainable=(not self.fix_biases),
                    initializer=tf.constant_initializer(0.0))

                if not self.fix_biases:
                    if self.bias_geom == 'hyp':
                        self.hyp_vars.append(self.b)
                    else:
                        self.eucl_vars.append(self.b)

                self.built = True

            new_h = self.one_rnn_transform(self.W, state, self.U, inputs, self.b)
            new_h = util.tf_hyp_non_lin(new_h, non_lin=self.non_lin, hyp_output=True, c=self.c_val)

        return new_h, new_h

################################################################################################


class HypGRU(tf.nn.rnn_cell.RNNCell):

    def __init__(self,
                 num_units,
                 inputs_geom,
                 bias_geom,
                 c_val,
                 non_lin,
                 fix_biases,
                 fix_matrices,
                 matrices_init_eye,
                 dtype):
        self._num_units = num_units
        self.c_val = c_val
        self.built = False
        self.__dtype = dtype
        self.non_lin = non_lin
        assert self.non_lin in ['id', 'relu', 'tanh', 'sigmoid']

        self.bias_geom = bias_geom
        self.inputs_geom = inputs_geom
        assert self.inputs_geom in ['eucl', 'hyp']
        assert self.bias_geom in ['eucl', 'hyp']

        self.fix_biases = fix_biases
        self.fix_matrices = fix_matrices
        if matrices_init_eye or self.fix_matrices:
            self.matrix_initializer = tf.initializers.identity()
        else:
            self.matrix_initializer = tf.contrib.layers.xavier_initializer()

        self.eucl_vars = []
        self.hyp_vars = []


    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units


    # Performs the hyperbolic version of the operation Wh + Ux + b.
    def one_rnn_transform(self, W, h, U, x, b):
        hyp_b = b
        if self.bias_geom == 'eucl':
            hyp_b = util.tf_exp_map_zero(b, self.c_val)

        W_otimes_h = util.tf_mob_mat_mul(W, h, self.c_val)
        U_otimes_x = util.tf_mob_mat_mul(U, x, self.c_val)
        Wh_plus_Ux = util.tf_mob_add(W_otimes_h, U_otimes_x, self.c_val)
        return util.tf_mob_add(Wh_plus_Ux, hyp_b, self.c_val)

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            if not self.built:
                inputs_shape = inputs.get_shape()
                print('Init GRU cell')
                if inputs_shape[1].value is None:
                    raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                                     % inputs_shape)
                input_depth = inputs_shape[1].value

                self.Wz = tf.get_variable('W_z', dtype= self.__dtype,
                                          shape=[self._num_units, self._num_units],
                                          trainable=(not self.fix_matrices),
                                          initializer=self.matrix_initializer)
                if not self.fix_matrices:
                    self.eucl_vars.append(self.Wz)
                self.Uz = tf.get_variable('U_z', dtype= self.__dtype,
                                          shape=[input_depth, self._num_units],
                                          trainable=(not self.fix_matrices),
                                          initializer=self.matrix_initializer)
                if not self.fix_matrices:
                    self.eucl_vars.append(self.Uz)
                self.bz = tf.get_variable('b_z', dtype= self.__dtype,
                                          shape=[1, self._num_units],
                                          trainable=(not self.fix_biases),
                                          initializer=tf.constant_initializer(0.0))
                if not self.fix_biases:
                    if self.bias_geom == 'hyp':
                        self.hyp_vars.append(self.bz)
                    else:
                        self.eucl_vars.append(self.bz)
                ###########################################

                self.Wr = tf.get_variable('W_r', dtype= self.__dtype,
                                          shape=[self._num_units, self._num_units],
                                          trainable=(not self.fix_matrices),
                                          initializer=self.matrix_initializer)
                if not self.fix_matrices:
                    self.eucl_vars.append(self.Wr)
                self.Ur = tf.get_variable('U_r', dtype= self.__dtype,
                                          shape=[input_depth, self._num_units],
                                          trainable=(not self.fix_matrices),
                                          initializer=self.matrix_initializer)
                if not self.fix_matrices:
                    self.eucl_vars.append(self.Ur)
                self.br = tf.get_variable('b_r', dtype= self.__dtype,
                                          shape=[1, self._num_units],
                                          trainable=(not self.fix_biases),
                                          initializer=tf.constant_initializer(0.0))
                if not self.fix_biases:
                    if self.bias_geom == 'hyp':
                        self.hyp_vars.append(self.br)
                    else:
                        self.eucl_vars.append(self.br)
                ###########################################

                self.Wh = tf.get_variable('W_h', dtype= self.__dtype,
                                          shape=[self._num_units, self._num_units],
                                          trainable=(not self.fix_matrices),
                                          initializer=self.matrix_initializer)
                if not self.fix_matrices:
                    self.eucl_vars.append(self.Wh)
                self.Uh = tf.get_variable('U_h', dtype= self.__dtype,
                                          shape=[input_depth, self._num_units],
                                          trainable=(not self.fix_matrices),
                                          initializer=self.matrix_initializer)
                if not self.fix_matrices:
                    self.eucl_vars.append(self.Uh)
                self.bh = tf.get_variable('b_h', dtype= self.__dtype,
                                          shape=[1, self._num_units],
                                          trainable=(not self.fix_biases),
                                          initializer=tf.constant_initializer(0.0))
                if not self.fix_biases:
                    if self.bias_geom == 'hyp':
                        self.hyp_vars.append(self.bh)
                    else:
                        self.eucl_vars.append(self.bh)
                ###########################################

                self.built = True

            hyp_x = inputs
            if self.inputs_geom == 'eucl':
                hyp_x = util.tf_exp_map_zero(inputs, self.c_val)

            z = util.tf_hyp_non_lin(self.one_rnn_transform(self.Wz, state, self.Uz, hyp_x, self.bz),
                                    non_lin='sigmoid',
                                    hyp_output=False,
                                    c = self.c_val)

            r = util.tf_hyp_non_lin(self.one_rnn_transform(self.Wr, state, self.Ur, hyp_x, self.br),
                                    non_lin='sigmoid',
                                    hyp_output=False,
                                    c = self.c_val)

            r_point_h = util.tf_mob_pointwise_prod(state, r, self.c_val)
            h_tilde = util.tf_hyp_non_lin(self.one_rnn_transform(self.Wh, r_point_h, self.Uh, hyp_x, self.bh),
                                          non_lin=self.non_lin,
                                          hyp_output=True,
                                          c=self.c_val)

            minus_h_oplus_htilde = util.tf_mob_add(-state, h_tilde, self.c_val)
            new_h = util.tf_mob_add(state,
                                    util.tf_mob_pointwise_prod(minus_h_oplus_htilde, z, self.c_val),
                                    self.c_val)
        return new_h, new_h

################################################################################################
