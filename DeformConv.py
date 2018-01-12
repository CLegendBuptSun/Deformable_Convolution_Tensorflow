"""
------------------------------------------------------------------
------------------------------------------------------------------
This is a tensorflow implementation of deformable convolution in
the paper Deformable Convolutional Networks.


by DouDou

2018-01-11
------------------------------------------------------------------
------------------------------------------------------------------

"""

import numpy as np
import tensorflow as tf


class DeformConv2D(object):
    """

    Definition of DeformConv2D class

    """

    def __init__(self, x_, ks, co, name, groups, trainable):
        """

        Initialization.


        Params:
        --- x_: Input of the deformable convolutional layer, a 4-D
                Tensor with shape [bsize, height, width, channel].
        --- ks: Value of the kernel size.
        --- co: Output channels (Amount of kernels).
        --- name: Name of the deformable convolution layer.
        --- groups: The amount of groups.
        --- trainable: Whether the weights are trainable or not.

        """

        self.x = x_

        self.ks = ks

        self.co = co

        # Number of kernel elements
        self.N = ks ** 2

        self.name = name

        self.groups = groups

        self.trainable = trainable

    def conv(self, x_, co, mode, relu=True, groups=1, stride=1):
        """

        Definition of the regular 2D convolutional layer.


        Params:
        --- x_: Input of the convolutional layer, a 4-D Tensor with
                shape [bsize, height, width, channel].
        --- co: Output channels (Amount of kernels).
        --- mode: Purpose of convolution, "feature" or "offset".
        --- relu: Whether to apply the relu non-linearity or not.
        --- groups: The amount of groups.
        --- stride: Value of stride when doing convolution.
        Return:
        --- layer_output: Output of the convolutional layer.

        """

        # Ensure the mode is valid
        assert mode in ["feature", "offset"]

        with tf.name_scope(self.name + "_" + mode):

            # Get the kernel size
            ks = self.ks

            # Get the input channel
            ci = x_.get_shape()[-1] / groups

            # Create the weights and biases
            if mode == "offset":

                with tf.variable_scope(self.name + "_offset"):

                    # In offset mode, the weights are zero initialized
                    weights = tf.get_variable(name="weights",
                                              shape=[ks, ks, ci, co],
                                              trainable=self.trainable,
                                              initializer=tf.zeros_initializer)

                    # Create the biases
                    biases = tf.get_variable(name="biases",
                                             shape=[co],
                                             trainable=self.trainable,
                                             initializer=tf.zeros_initializer)

            else:

                with tf.variable_scope(self.name):

                    weights = tf.get_variable(name="weights",
                                              shape=[ks, ks, ci, co],
                                              trainable=self.trainable)

                    # Create the biases
                    biases = tf.get_variable(name="biases",
                                             shape=[co],
                                             trainable=self.trainable,
                                             initializer=tf.zeros_initializer)

            # Define function for convolution calculation
            def conv2d(i_, w_):

                return tf.nn.conv2d(i_, w_, [1, stride, stride, 1], padding="SAME")

            # If we don't need to divide this convolutional layer
            if groups == 1:

                layer_output = conv2d(x_, weights)

            # If we need to divide this convolutional layer
            else:

                # Split the input and weights
                group_inputs = tf.split(x_, groups, 3, name="split_input")

                group_weights = tf.split(weights, groups, 3, name="split_weight")

                group_outputs = [conv2d(i, w) for i, w in zip(group_inputs, group_weights)]

                # Concatenate the groups
                layer_output = tf.concat(group_outputs, 3)

            # Add the biases
            layer_output = tf.nn.bias_add(layer_output, biases)

            if relu:

                # Nonlinear process
                layer_output = tf.nn.relu(layer_output)

            return layer_output

    def infer(self):
        """

        Function for deformable convolution.


        Return:
        --- layer_output: Output of the deformable convolutional layer.

        """

        with tf.name_scope(self.name):

            # Get the layer input
            x = self.x[:, :, :, :]

            # Get the kernel size.
            ks = self.ks

            # Get the number of kernel elements.
            N = self.N

            # Get the shape of the layer input.
            bsize, h, w, c = x.get_shape().as_list()

            # Get the offset, with shape [bsize, h, w, 2N].
            offset = self.conv(x, 2 * N, "offset", relu=False)

            # Get the data type of offset
            dtype = offset.dtype

            # pn ([1, 1, 1, 2N]) contains the locations in the kernel.
            pn = self.get_pn(dtype)

            # p0 ([1, h, w, 2N]) contains the location of each pixel on
            # the output feature map.
            p0 = self.get_p0([bsize, h, w, c], dtype)

            # p ([bsize, h, w, 2N]) contains the sample locations on the
            # input feature map of each pixel on the output feature map.
            p = p0 + pn + offset

            # Reshape p to [bsize, h, w, 2N, 1, 1].
            p = tf.reshape(p, [bsize, h, w, 2 * N, 1, 1])

            # q ([h, w, 2]) contains the location of each pixel on the
            # output feature map.
            q = self.get_q([bsize, h, w, c], dtype)

            # Get the bilinear interpolation kernel G ([bsize, h, w, N, h, w])
            gx = tf.maximum(1 - tf.abs(p[:, :, :, :N, :, :] - q[:, :, 0]), 0)

            gy = tf.maximum(1 - tf.abs(p[:, :, :, N:, :, :] - q[:, :, 1]), 0)

            G = gx * gy

            # Reshape G to [bsize, h*w*N, h*w]
            G = tf.reshape(G, [bsize, h * w * N, h * w])

            # Reshape x to [bsize, h*w, c]
            x = tf.reshape(x, [bsize, h*w, c])

            # Get x_offset
            x_offset = tf.reshape(tf.matmul(G, x), [bsize, h, w, N, c])

            # Reshape x_offset to [bsize, h*kernel_size, w*kernel_size, c]
            x_offset = self.reshape_x_offset(x_offset, ks)

            # Get the output of the deformable convolutional layer
            layer_output = self.conv(x_offset, self.co, "feature", groups=self.groups, stride=ks)

            return layer_output

    def get_pn(self, dtype):
        """

        Function to get pn.


        Params:
        --- dtype: Data type of pn.
        Return:
        --- pn: A 4-D Tensor with shape [1, 1, 1, 2N], which contains
                the locations in the kernel.

        """

        # Get the kernel size
        ks = self.ks

        pn_x, pn_y = np.meshgrid(range(-(ks-1)/2, (ks-1)/2+1), range(-(ks-1)/2, (ks-1)/2+1), indexing="ij")

        # The shape of pn is [2N,], order [x1, x2, ..., y1, y2, ...]
        pn = np.concatenate((pn_x.flatten(), pn_y.flatten()))

        # Reshape pn to [1, 1, 1, 2N]
        pn = np.reshape(pn, [1, 1, 1, 2 * self.N])

        # Convert pn to TF Tensor
        pn = tf.constant(pn, dtype)

        return pn

    def get_p0(self, x_size, dtype):
        """

        Function to get p0.


        Params:
        --- x_size: Size of the input feature map.
        --- dtype: Data type of p0.
        Return:
        --- p0: A 4-D Tensor with shape [1, h, w, 2N], which contains
                the locations of each pixel on the output feature map.

        """

        # Get the shape of input feature map.
        bsize, h, w, c = x_size

        p0_x, p0_y = np.meshgrid(range(0, h), range(0, w), indexing="ij")

        p0_x = p0_x.flatten().reshape(1, h, w, 1).repeat(self.N, axis=3)

        p0_y = p0_y.flatten().reshape(1, h, w, 1).repeat(self.N, axis=3)

        p0 = np.concatenate((p0_x, p0_y), axis=3)

        # Convert p0 to TF Tensor
        p0 = tf.constant(p0, dtype)

        return p0

    def get_q(self, x_size, dtype):
        """

        Function to get q.


        Params:
        --- x_size: Size of the input feature map.
        --- dtype: Data type of q.
        Return:
        --- q: A 3-D Tensor with shape [h, w, 2], which contains the
               locations of each pixel on the output feature map.

        """

        # Get the shape of input feature map.
        bsize, h, w, c = x_size

        q_x, q_y = np.meshgrid(range(0, h), range(0, w), indexing="ij")

        q_x = q_x.flatten().reshape(h, w, 1)

        q_y = q_y.flatten().reshape(h, w, 1)

        q = np.concatenate((q_x, q_y))

        # Convert q to TF Tensor
        q = tf.constant(q, dtype)

        return q

    @staticmethod
    def reshape_x_offset(x_offset, ks):
        """
        Function to reshape x_offset.


        Params:
        --- x_offset: A 5-D Tensor with shape [bsize, h, w, N, c].
        --- ks: The value of kernel size.
        Return:
        --- x_offset: A 4-D Tensor with shape [bsize, h*ks, w*ks, c].

        """

        # Get the shape of x_offset.
        bsize, h, w, N, c = x_offset.get_shape().as_list()

        # Get the new_shape
        new_shape = [bsize, h, w * ks, c]

        x_offset = [tf.reshape(x_offset[:, :, :, s:s+ks, :], new_shape) for s in range(0, N, ks)]

        x_offset = tf.concat(x_offset, axis=2)

        # Reshape to final shape [bsize, h*ks, w*ks, c]
        x_offset = tf.reshape(x_offset, [bsize, h * ks, w * ks, c])

        return x_offset
