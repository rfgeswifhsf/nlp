import tensorflow as tf
import  numpy as np

A = [[1, 3, 4, 5, 6]]
B = [[1, 3, 4], [2, 4, 1]]
C = [[[16, 6, 3], [4, 6, 5]], [[8, 7, 9], [11, 12, 10]]]

with tf.Session() as sess:
    print(np.shape(A))
    print(sess.run(tf.argmax(A, 0)))
    print(sess.run(tf.argmax(A, 1)))
    print(sess.run(tf.argmax(B, 0)))
    print(sess.run(tf.argmax(B, 1)))
    print(sess.run(tf.argmax(C, 0)))
    print(sess.run(tf.argmax(C, 1)))
    print(sess.run(tf.argmax(C, 2)))

'''
轴012
轴0：列，
轴1：行
对于三维：
轴0：依然可理解为列。三维矩阵内同等位置的向量相互比较，c中[16, 6, 3]与[8, 7, 9]比较
轴1：依然可理解为行。被第二层包裹的矩阵之间个向量相互比较 [16, 6, 3]与 [4, 6, 5]
轴2：为矩阵更深以及比较，即为各个向量内部之间的比较。
'''
