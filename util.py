import logging
import tensorflow as tf
import numpy as np
from numpy import linalg as LA
from numpy import random as np_random
import os
import random

PROJ_EPS = 1e-5
EPS = 1e-15
MAX_TANH_ARG = 15.0


def tf_project_hyp_vecs(x, c):
    # Projection op. Need to make sure hyperbolic embeddings are inside the unit ball.
    return tf.clip_by_norm(t=x, clip_norm=(1. - PROJ_EPS) / np.sqrt(c), axes=[1])

######################## x,y have shape [batch_size, emb_dim] in all tf_* functions ################

# Real x, not vector!
def tf_atanh(x):
    return tf.atanh(tf.minimum(x, 1. - EPS)) # Only works for positive real x.

# Real x, not vector!
def tf_tanh(x):
   return tf.tanh(tf.minimum(tf.maximum(x, -MAX_TANH_ARG), MAX_TANH_ARG))


def tf_dot(x, y):
    return tf.reduce_sum(x * y, axis=1, keepdims=True)

def tf_norm(x):
    return tf.norm(x, axis = 1, keepdims=True)

#########################
def tf_mob_add(u, v, c):
    v = v + EPS
    tf_dot_u_v = 2. * c * tf_dot(u, v)
    tf_norm_u_sq = c * tf_dot(u,u)
    tf_norm_v_sq = c * tf_dot(v,v)
    denominator = 1. + tf_dot_u_v + tf_norm_v_sq * tf_norm_u_sq
    result = (1. + tf_dot_u_v + tf_norm_v_sq) / denominator * u + (1. - tf_norm_u_sq) / denominator * v
    return tf_project_hyp_vecs(result, c)

def mob_add(u, v, c):
    numerator = (1.0 + 2.0 * c * np.dot(u,v) + c * LA.norm(v)**2) * u + (1.0 - c * LA.norm(u)**2) * v
    denominator = 1.0 + 2.0 * c * np.dot(u,v) + c**2 * LA.norm(v)**2 * LA.norm(u)**2
    return numerator / denominator


#########################
def tf_poinc_dist_sq(u, v, c):
    sqrt_c = np.sqrt(c)
    m = tf_mob_add(-u, v, c) + EPS
    atanh_x = np.sqrt(c) * tf_norm(m)
    dist_poincare = 2. / sqrt_c * tf_atanh(atanh_x)
    return dist_poincare ** 2

def poinc_dist_sq(u, v, c):
    sqrt_c = np.sqrt(c)
    atanh_x = sqrt_c * LA.norm(mob_add(-u, v, c))
    dist_poincare = 2.0 / sqrt_c * np.arctanh(atanh_x)
    return dist_poincare ** 2


#########################
def tf_euclid_dist_sq(u, v):
    return tf.reduce_sum(tf.square(u - v), axis=1, keepdims=True)

def euclid_dist_sq(u, v):
    return LA.norm(u - v)


#########################
def tf_mob_scalar_mul(r, v, c):
    v = v + EPS
    norm_v = tf_norm(v)
    nomin = tf_tanh(r * tf_atanh(np.sqrt(c) * norm_v))
    result= nomin / (np.sqrt(c) * norm_v) * v
    return tf_project_hyp_vecs(result, c)

def mob_scalar_mul(r, v, c):
    norm_v = LA.norm(v)
    nomin = np.tanh(r * np.arctanh(np.sqrt(c) * norm_v)) * v
    return nomin / (np.sqrt(c) * norm_v)


#########################
def tf_lambda_x(x, c):
    return 2. / (1 - c * tf_dot(x,x))

def lambda_x(x, c):
    return 2. / (1 - c * LA.norm(x)**2)


#########################
def unit_speed_geo(x, v, t, c):
    second_term = np.tanh(np.sqrt(c) * t / 2) / (np.sqrt(c) * LA.norm(v)) * v
    return mob_add(x, second_term, c)

def exp_map_x(x, v, c):
    second_term = np.tanh(np.sqrt(c) * lambda_x(x, c) * LA.norm(v) / 2) / (np.sqrt(c) * LA.norm(v)) * v
    return mob_add(x, second_term, c)

def log_map_x(x, y, c):
    diff = mob_add(-x, y, c)
    lam = lambda_x(x, c)
    return 2. / (np.sqrt(c) * lam) * np.arctanh(np.sqrt(c) * LA.norm(diff)) / (LA.norm(diff)) * diff

def tf_exp_map_x(x, v, c):
    v = v + EPS # Perturbe v to avoid dealing with v = 0
    norm_v = tf_norm(v)
    second_term = (tf_tanh(np.sqrt(c) * tf_lambda_x(x, c) * norm_v / 2) / (np.sqrt(c) * norm_v)) * v
    return tf_mob_add(x, second_term, c)

def tf_log_map_x(x, y, c):
    diff = tf_mob_add(-x, y, c) + EPS
    norm_diff = tf_norm(diff)
    lam = tf_lambda_x(x, c)
    return ( ( (2. / np.sqrt(c)) / lam) * tf_atanh(np.sqrt(c) * norm_diff) / norm_diff) * diff



def tf_exp_map_zero(v, c):
    v = v + EPS # Perturbe v to avoid dealing with v = 0
    norm_v = tf_norm(v)
    result = tf_tanh(np.sqrt(c) * norm_v) / (np.sqrt(c) * norm_v) * v
    return tf_project_hyp_vecs(result, c)

def tf_log_map_zero(y, c):
    diff = y + EPS
    norm_diff = tf_norm(diff)
    return 1. / np.sqrt(c) * tf_atanh(np.sqrt(c) * norm_diff) / norm_diff * diff


#########################
def mob_mat_mul(M, x, c):
    Mx = M.dot(x)
    MX_norm = LA.norm(Mx)
    x_norm = LA.norm(x)
    return 1. / np.sqrt(c) * np.tanh(MX_norm / x_norm * np.arctanh(np.sqrt(c) * x_norm)) / MX_norm * Mx

def tf_mob_mat_mul(M, x, c):
    x = x + EPS
    Mx = tf.matmul(x, M) + EPS
    MX_norm = tf_norm(Mx)
    x_norm = tf_norm(x)
    result = 1. / np.sqrt(c) * tf_tanh(MX_norm / x_norm * tf_atanh(np.sqrt(c) * x_norm)) / MX_norm * Mx
    return tf_project_hyp_vecs(result, c)



# x is hyperbolic, u is Euclidean. Computes diag(u) \otimes x.
def tf_mob_pointwise_prod(x, u, c):
    x = x + EPS
    Mx = x * u + EPS
    MX_norm = tf_norm(Mx)
    x_norm = tf_norm(x)
    result = 1. / np.sqrt(c) * tf_tanh(MX_norm / x_norm * tf_atanh(np.sqrt(c) * x_norm)) / MX_norm * Mx
    return tf_project_hyp_vecs(result, c)

#########################
def riemannian_gradient_c(u, c):
    return ((1. - c * tf_dot(u,u)) ** 2) / 4.0


#########################
def tf_eucl_non_lin(eucl_h, non_lin):
    if non_lin == 'id':
        return eucl_h
    elif non_lin == 'relu':
        return tf.nn.relu(eucl_h)
    elif non_lin == 'tanh':
        return tf.tanh(eucl_h)
    elif non_lin == 'sigmoid':
        return tf.nn.sigmoid(eucl_h)
    return eucl_h

# Applies a non linearity sigma to a hyperbolic h using: exp_0(sigma(log_0(h)))
def tf_hyp_non_lin(hyp_h, non_lin, hyp_output, c):
    if non_lin == 'id':
        if hyp_output:
            return hyp_h
        else:
            return tf_log_map_zero(hyp_h, c)

    eucl_h = tf_eucl_non_lin(tf_log_map_zero(hyp_h, c), non_lin)

    if hyp_output:
        return tf_exp_map_zero(eucl_h, c)
    else:
        return eucl_h


####################################################################################################
####################################################################################################
####################################### Unit tests #################################################
####################################################################################################
####################################################################################################
def mobius_addition_left_cancelation_test():
    for i in range(0, 10000):
        a = np.random.uniform(low=-0.01, high=0.01, size=1)
        b = np.random.uniform(low=-0.01, high=0.01, size=1)

        c = random.random()
        res = mob_add(-a, mob_add(a, b, c=c), c=c)
        diff = np.sum(np.abs(b - res))
        if diff > 1e-10:
            print('Invalid :/')
            print('b: ')
            print(b)
            print('res: ')
            print(res)
            exit()

    print('Test left cancelation passed!')


def mobius_addition_cancel_test():
    for i in range(0, 10000):
        a = np.random.uniform(low=-0.01, high=0.01, size=10)

        res = mob_add(-a, a, c=random.random())
        diff = np.sum(np.abs(res))
        if diff > 1e-10:
            print('Invalid :/')
            print('res: ')
            print(res)
            exit()

    print('Test -a + a passed!')


def mobius_addition_2a_test():
    for i in range(0, 10000):
        a = np.random.uniform(low=-0.01, high=0.01, size=10)

        res1 = mob_add(a, a, c=1.0)
        res2 = 2.0 / (1.0 + np.dot(a, a)) * a
        diff = np.sum(np.abs(res1 - res2))
        if diff > 1e-10:
            print('Invalid :/')
            print('res1: ')
            print(res1)
            print('res2: ')
            print(res2)
            exit()

    print('Test a+a passed!')


def mobius_addition_poinc_dist_test():
    for i in range(0, 10000):
        a = np.random.uniform(low=0.0, high=0.01, size=10)
        b = np.random.uniform(low=0.0, high=0.01, size=10)

        res1 = poinc_dist_sq(a, b, c=1.0)
        res2 = 2 * np.arctanh(np.linalg.norm(mob_add(-a, b, c=1.0)))
        diff = np.sum(np.abs(res1 - res2**2))
        if diff > 1e-10:
            print('Test 4 FAILED at trial %d :/' % i)
            print('res1: ')
            print(res1)
            print('res2: ')
            print(res2)
            print('2xres2: ')
            print(2 * res2)
            print('2xres2 - res1')
            print(2 * res2 - res1)
            return

    print('Test poinc dist - mobius passed!')


def mobius_addition_zero_b_test():
    for i in range(0, 10000):
        a = np.zeros(10)
        b = np.random.uniform(low=-0.01, high=0.01, size=10)

        res = mob_add(a, b, c=1.0)
        diff = np.sum(np.abs(res - b))
        if diff > 1e-10:
            print('Test 5 FAILED at trial %d :/' % i)
            print('res: ')
            print(res)
            print('b: ')
            print(b)
            exit()

    print('Test 0 + b passed!')


def mobius_addition_negative_test():
    for i in range(0, 10000):
        a = np.random.uniform(low=-0.01, high=0.01, size=10)
        b = np.random.uniform(low=-0.01, high=0.01, size=10)

        c = random.random()
        res1 = mob_add(-a, -b, c)
        res2 = -mob_add(a, b, c)
        diff = np.sum(np.abs(res1 - res2))

        if diff > 1e-10:
            print('Test 6 FAILED at trial %d :/' % i)
            print('res1: ')
            print(res1)
            print('res2: ')
            print(res2)
            exit()

    print('Test a+b = -a + -b passed!')


def mobius_addition_infinity_test():
    for i in range(0, 10000):
        a = np.random.uniform(low=-0.01, high=0.01, size=10)
        b = np.random.uniform(low=-0.01, high=0.01, size=10)

        a = a / LA.norm(a)

        res = mob_add(a, b, c=1.0)
        diff = LA.norm(a - res)

        if diff > 1e-10:
            print('Test 7 FAILED at trial %d :/' % i)
            print('res: ')
            print(res)
            print('diff: ')
            print(diff)
            exit()

        res = mob_add(b, a, c=1.0)
        diff = np.abs(1 - LA.norm(res))

        if diff > 1e-10:
            print('Test 7 FAILED at trial %d :/' % i)
            print('res: ')
            print(res)
            print('diff: ')
            print(diff)
            exit()


    print('Test mob add at infinity passed!')


def mobius_test_TF():
    emb_dim = 20
    c = 1

    bs = 1
    r = np_random.random() * 10

    v1 = tf.placeholder(tf.float64, shape=[bs, emb_dim])
    v2 = tf.placeholder(tf.float64, shape=[bs, emb_dim])
    M = tf.placeholder(tf.float64, shape=[emb_dim, 5])

    v1_instance = np_random.uniform(-.5, .5, (bs, emb_dim)).astype(np.float64)
    v2_instance = np_random.uniform(-.5, .5, (bs, emb_dim)).astype(np.float64)

    v1_instance = v1_instance * 0.59999 / LA.norm(v1_instance)
    v2_instance = v2_instance * 0.99 / LA.norm(v2_instance)

    M_instance = np.random.rand(emb_dim, 5).astype(np.float64)

    for c_pow in range(15):
        c = 10 ** (- c_pow)

        poinc_dist_op = tf_poinc_dist_sq(v1, v2, c)
        eucl_dist_op = tf_euclid_dist_sq(v1, v2)
        mob_dif_op = tf_mob_add(-v1, v2, c)
        mob_scalar_mul_op = tf_mob_scalar_mul(r, v1, c)
        lambda_x_op = tf_lambda_x(v1 * 0.5, c)
        exp_map_x_op = tf_exp_map_x(v1, v2, c)
        log_map_x_op = tf_log_map_x(v1, v2, c)
        mat_mul_op = tf_mob_mat_mul(M, v1, c)

        with tf.Session() as sess:
            mat_mul_v, exp_map_x_v, log_map_x_v, lambda_x_v, mob_scalar_mul_v, mob_dif_v, poinc_dist_v, eucl_dist_v = \
                sess.run([mat_mul_op, exp_map_x_op, log_map_x_op, lambda_x_op, mob_scalar_mul_op, mob_dif_op, poinc_dist_op, eucl_dist_op],
                         feed_dict={
                             v1: v1_instance,
                             v2: v2_instance,
                             M: M_instance
                         })

        assert LA.norm(mat_mul_v - mob_mat_mul(M_instance.T, v1_instance.reshape([-1]), c)) < 1e-8

        assert LA.norm(exp_map_x_v - exp_map_x(v1_instance.reshape([-1]), v2_instance.reshape([-1]), c)) < 1e-8
        assert LA.norm(log_map_x_v - log_map_x(v1_instance.reshape([-1]), v2_instance.reshape([-1]), c)) < 1e-8

        assert abs(lambda_x_v - lambda_x(v1_instance.reshape([-1]) * 0.5, c)) < 1e-10
        assert abs(poinc_dist_v - poinc_dist_sq(v1_instance.reshape([-1]), v2_instance.reshape([-1]), c)) < 1e-5

    print('Test TF passed!')


def mobius_unit_speed_geo_test():

    emb_dim = 5
    c = 0.76

    x = np_random.uniform(-.5, .5, (emb_dim)).astype(np.float64)
    v = np_random.uniform(-.5, .5, (emb_dim)).astype(np.float64)

    x = x * 0.54321 / LA.norm(x)
    v = v / (LA.norm(v) * lambda_x(x, c))

    t = 1e-6
    d = (- unit_speed_geo(x, v, 0, c) + unit_speed_geo(x, v, t, c)) / t

    assert LA.norm(d - v) < 1e-5
    print('Test unit speed geodesic passed!')


def mobius_exp_map_test():
    emb_dim = 5
    c = 0.76

    x = np_random.uniform(-.5, .5, (emb_dim)).astype(np.float64)
    v = np_random.uniform(-.5, .5, (emb_dim)).astype(np.float64)

    x = x * 0.54321 / LA.norm(x)

    assert LA.norm(log_map_x(x, exp_map_x(x, v, c), c) - v ) < 1e-5

    r = np_random.random() * 10
    assert LA.norm(exp_map_x(0, r * log_map_x(0, x, c), c) - mob_scalar_mul(r, x, c)) < 1e-8

    print('Test exp map passed!')

def mobius_mat_mul_test():
    M = np.random.rand(5, 8)
    x = np.random.rand(8)
    x = x / LA.norm(x) * 0.789
    c = 1.0
    assert LA.norm(mob_mat_mul(M, x, c) - exp_map_x(0, M.dot(log_map_x(0, x, c)), c)) < 1e-5

    for i in range(10):
        c = random.random()
        assert LA.norm(mob_mat_mul(M, x, c) - exp_map_x(0, M.dot(log_map_x(0, x, c)), c)) < 1e-5
        assert LA.norm(mob_mat_mul(M, x, 1e-10) - M.dot(x)) < 1e-5

        M_prime = np.random.rand(7, 5)
        assert LA.norm(mob_mat_mul(M_prime.dot(M), x, c) - mob_mat_mul(M_prime, mob_mat_mul(M,x,c),c)) < 1e-5

        r = random.random() * 10
        assert LA.norm(mob_mat_mul(r * M, x, c) - mob_scalar_mul(r, mob_mat_mul(M,x,c),c)) < 1e-5

        assert LA.norm(mob_mat_mul(M, x, c) / LA.norm(mob_mat_mul(M, x, c)) - M.dot(x) / LA.norm(M.dot(x))) < 1e-5

    print('Mobius mat mul test passed!')


def run_all_unit_tests():
    mobius_unit_speed_geo_test()
    mobius_mat_mul_test()
    mobius_exp_map_test()
    mobius_addition_left_cancelation_test()
    mobius_addition_cancel_test()
    mobius_addition_2a_test()
    mobius_addition_poinc_dist_test()
    mobius_addition_zero_b_test()
    mobius_addition_negative_test()
    mobius_addition_infinity_test()
    mobius_test_TF()

# run_all_unit_tests()


####################################################################################################

def setup_logger(name_logfile, logs_dir, also_stdout=False):
    name_logfile = name_logfile.replace(';', '#')
    name_logfile = name_logfile.replace(':', '_')
    logger = logging.getLogger(name_logfile)
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    fileHandler = logging.FileHandler(os.path.join(logs_dir, name_logfile), mode='w')
    fileHandler.setFormatter(formatter)
    if also_stdout:
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(fileHandler)
    if also_stdout:
        logger.addHandler(streamHandler)
    return logger






