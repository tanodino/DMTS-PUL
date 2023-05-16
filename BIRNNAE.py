'''
  Copyright (C) 2023 Dino Ienco

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; see the file COPYING. If not, write to the
  Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
'''


import tensorflow as tf


class RNNAE(tf.keras.Model):
    def __init__(self, filters, outputDim, dropout_rate = 0.0, name='RNNAE', **kwargs):
        # chiamata al costruttore della classe padre, Model
        super(RNNAE, self).__init__(name=name, **kwargs)
        self.encoderR = tf.keras.layers.LSTM(filters, go_backwards=True)
        self.encoder = tf.keras.layers.LSTM(filters)

        self.decoder = tf.keras.layers.LSTM(filters, return_sequences=True)
        self.decoder2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=outputDim, activation=None))

        self.decoderR = tf.keras.layers.LSTM(filters, return_sequences=True)
        self.decoder2R = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=outputDim, activation=None))

    #@tf.function
    def call(self, inputs, training=False):
        t = inputs.get_shape()
        enc = self.encoder(inputs)
        emb = enc
        seq_emb = tf.keras.layers.RepeatVector(t[1])(emb)
        dec = self.decoder(seq_emb)
        dec = self.decoder2(dec)


        encR = self.encoderR(inputs)
        embR = encR
        seq_embR = tf.keras.layers.RepeatVector(t[1])(embR)
        decR = self.decoderR(seq_embR)
        decR = self.decoder2R(decR)
        decR = tf.reverse(decR, axis=[1])

        return dec, decR, tf.concat((emb,embR),axis=1)

        #(dec+decR)/2, tf.concat((emb,embR),axis=1), tf.concat((emb,embR),axis=1), tf.concat((emb,embR),axis=1)
