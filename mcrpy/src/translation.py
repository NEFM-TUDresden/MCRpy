import tensorflow as tf

def translate_2D(x: tf.Tensor, translation_x: int, translation_y: int) -> tf.Tensor:
    if len(tf.shape(x)) == 3:
        x = tf.expand_dims(x, 0)
    tf.assert_equal(len(tf.shape(x)), 4)
    batch_size = tf.shape(x)[0]
    image_size = tf.shape(x)[1:3]
    grid_x0 = tf.expand_dims(tf.range(image_size[0], dtype=tf.int32), 0) + tf.zeros([batch_size, 1], dtype=tf.int32)
    grid_y0 = tf.expand_dims(tf.range(image_size[1], dtype=tf.int32), 0) + tf.zeros([batch_size, 1], dtype=tf.int32)
    grid_x = tf.expand_dims(tf.range(image_size[0], dtype=tf.int32), 0) + translation_x + image_size[0]
    grid_y = tf.expand_dims(tf.range(image_size[1], dtype=tf.int32), 0) + translation_y + image_size[1]
    grid_x = tf.gather_nd(tf.concat([grid_x0,grid_x0,grid_x0], 1), tf.expand_dims(grid_x, -1), batch_dims=1)
    grid_y = tf.gather_nd(tf.concat([grid_y0,grid_y0,grid_y0], 1), tf.expand_dims(grid_y, -1), batch_dims=1)
    x = tf.gather_nd(x, tf.expand_dims(grid_x, -1), batch_dims=1)
    x = tf.transpose(tf.gather_nd(tf.transpose(x, [0, 2, 1, 3]), tf.expand_dims(grid_y, -1), batch_dims=1), [0, 2, 1, 3])
    return x
