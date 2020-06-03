
# coding: utf-8

# In[ ]:


import numpy as np
import deep_laa_support as dls
import random
import sys
import tensorflow as tf
from sklearn.cluster import KMeans

# read data
# filename = 'default_file'
data_all = np.load(filename +'.npz')
print('File ' + filename + '.npz ' 'loaded.')
user_labels = data_all['user_labels']
true_labels = data_all['true_labels']
category_size = data_all['category_num']
source_num = data_all['source_num']
feature = data_all['feature']
_, feature_size = np.shape(feature)
n_samples, _ = np.shape(true_labels)

#================= basic parameters =====================
# define batch size (use all samples in one batch)
batch_size = n_samples # n_samples
cluster_num = 200
T = 1 # mc_sampling_times

if np.max(feature) <= 1 and np.min(feature) >= 0:
    flag_node_type = 'Bernoulli'
else:
    flag_node_type = 'Gaussian'   
print(flag_node_type + ' output nodes are used.')

#================= encoder q(y|l) and q(h|x) =====================
with tf.name_scope('encoder'):
    #================= q(y|l) =====================
    # define input l (source label vectors)
    input_size = source_num * category_size
    with tf.variable_scope('q_yl'):
        l = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_size], name='l_input')
        pi_yl, weights_yl, biases_yl = dls.LAA_encoder(l, batch_size, source_num, category_size)
    # loss: cross entropy between y_classifier and y_target for pre-training classifier
    with tf.variable_scope('q_yl'):
        pi_yl_target = tf.placeholder(dtype=tf.float32, shape=[batch_size, category_size], name='pi_yl_target')
        loss_yl = dls.LAA_loss_classifier(pi_yl, pi_yl_target)
    # optimizier
    learning_rate_yl = 0.01
    optimizer_pre_train_yl = tf.train.AdamOptimizer(learning_rate=learning_rate_yl).minimize(loss_yl)

    #================= q(h|x) =====================
    h1_size_encoder = int(np.floor(feature_size/2.0))
    h2_size_encoder = 100
    embedding_size = 40
    h1_size_decoder = 100
    h2_size_decoder = int(np.floor(feature_size/2.0))
    with tf.variable_scope('q_hx'):
        x = tf.placeholder(dtype=tf.float32, shape=[batch_size, feature_size], name='x_input')
        # mu_hx[batch_size, embedding_size]
        # sigma_hx[batch_size, embedding_size]
        with tf.variable_scope('feature_encoder_h1'):
            _h1_encoder, w1_encoder, b1_encoder = dls.full_connect_relu_BN(x, [feature_size, h1_size_encoder])
        with tf.variable_scope('feature_encoder_h2'):
            _h2_encoder, w2_encoder, b2_encoder = dls.full_connect_relu_BN(_h1_encoder, [h1_size_encoder, h2_size_encoder])
        with tf.variable_scope('feature_encoder_mu'):
            mu_hx, w_mu_encoder, b_mu_encoder = dls.full_connect(_h2_encoder, [h2_size_encoder, embedding_size])
        with tf.variable_scope('feature_encoder_sigma'):
            sigma_hx, w_sigma_encoder, b_sigma_encoder = dls.full_connect(_h2_encoder, [h2_size_encoder, embedding_size])
        # mu_hx, sigma_hx = dls.vae_encoder(x, feature_size, h1_size_encoder, h2_size_encoder, embedding_size)
        # embedding_h[batch_size, T, embedding_size]
        embedding_h = tf.reshape(mu_hx, [batch_size, 1, embedding_size])             + tf.reshape(sigma_hx, [batch_size, 1, -1])             * tf.random_normal(shape=[batch_size, T, embedding_size], mean=0, stddev=1, dtype=tf.float32)

    with tf.variable_scope('q_hx_AE'):
        # x_reconstr, _, _ = dls.vae_decoder(mu_hx, embedding_size, h1_size_decoder, h2_size_decoder, feature_size)
        with tf.variable_scope('feature_decoder_h1'):
            _h1_decoder, w1_decoder, b1_decoder = dls.full_connect_relu_BN(mu_hx, [embedding_size, h1_size_decoder])
        with tf.variable_scope('feature_decoder_h2'):
            _h2_decoder, w2_decoder, b2_decoder = dls.full_connect_relu_BN(_h1_decoder, [h1_size_decoder, h2_size_decoder])
        with tf.variable_scope('feature_decoder_rho'):
            if flag_node_type == 'Bernoulli':
                x_reconstr, w_rho_decoder, b_rho_decoder = dls.full_connect_sigmoid(_h2_decoder, [h2_size_decoder, feature_size])
            elif flag_node_type == 'Gaussian':
                x_reconstr, w_rho_decoder, b_rho_decoder = dls.full_connect(_h2_decoder, [h2_size_decoder, feature_size])
        
        # Bernoulli
        loss_cross_entropy_AE = -tf.reduce_mean(tf.reduce_sum(x*tf.log(1e-10+x_reconstr) + (1.0-x)*tf.log(1e-10+(1.0-x_reconstr)), -1))
        # Gaussian
        loss_square_AE = 0.5 * tf.reduce_mean(tf.square(x_reconstr - x))
        constraint_w_AE = 0.5 * (tf.reduce_mean(tf.square(w1_encoder)) + tf.reduce_mean(tf.square(b1_encoder))
            + tf.reduce_mean(tf.square(w2_encoder)) + tf.reduce_mean(tf.square(b2_encoder))
            + tf.reduce_mean(tf.square(w_mu_encoder)) + tf.reduce_mean(tf.square(b_mu_encoder))
            + tf.reduce_mean(tf.square(w1_decoder)) + tf.reduce_mean(tf.square(b1_decoder))
            + tf.reduce_mean(tf.square(w2_decoder)) + tf.reduce_mean(tf.square(b2_decoder))
            + tf.reduce_mean(tf.square(w_rho_decoder)) + tf.reduce_mean(tf.square(b_rho_decoder)))
        if flag_node_type == 'Bernoulli':
            loss_AE = loss_cross_entropy_AE                 + constraint_w_AE
        elif flag_node_type == 'Gaussian':
            loss_AE = loss_square_AE                 + constraint_w_AE
        learning_rate_AE = 0.02
        optimizer_AE = tf.train.AdamOptimizer(learning_rate=learning_rate_AE).minimize(loss_AE)

    #================= p(x|h) =====================
    with tf.variable_scope('q_hx_AE'):
        with tf.variable_scope('feature_decoder_h1', reuse=True):
            _h_VAE = tf.reshape(embedding_h, [-1, embedding_size])
            _h1_decoder_VAE, _, _ = dls.full_connect_relu_BN(_h_VAE, [embedding_size, h1_size_decoder])
        with tf.variable_scope('feature_decoder_h2', reuse=True):
            _h2_decoder_VAE, _, _ = dls.full_connect_relu_BN(_h1_decoder_VAE, [h1_size_decoder, h2_size_decoder])
        with tf.variable_scope('feature_decoder_rho', reuse=True):
            if flag_node_type == 'Bernoulli':
                mu_xh, _, _ = dls.full_connect_sigmoid(_h2_decoder_VAE, [h2_size_decoder, feature_size])
            elif flag_node_type == 'Gaussian':
                mu_xh, _, _ = dls.full_connect(_h2_decoder_VAE, [h2_size_decoder, feature_size])
            mu_xh = tf.reshape(mu_xh, [batch_size, T, feature_size])
            
    print('Encoders are constructed.')
    
#================= decoder p(l|y), p(x|h), p(y|z), p(h|z) and p(z) =====================
with tf.name_scope('decoder'):
    #================= p(l|y) =====================
    with tf.variable_scope('p_ly'):
        # pi_ly[category_size, 1, source_num*category_size]
        pi_ly, weights_ly, biases_ly = dls.LAA_decoder(source_num, category_size)

        constraint_w_LAA = 0.5 * (tf.reduce_mean(tf.square(weights_ly)) + tf.reduce_mean(tf.square(biases_ly))
            + tf.reduce_mean(tf.square(weights_yl)) + tf.reduce_mean(tf.square(biases_yl)))
        
    #================= p(y|z) =====================
    with tf.variable_scope('p_yz'):
        # pi_yz[cluster_num, category_size]
        _pi_yz = tf.get_variable('pi_yz', dtype=tf.float32, 
                                initializer=tf.random_normal(shape=[cluster_num, category_size], mean=0, stddev=1, dtype=tf.float32))
        __pi_yz = tf.exp(_pi_yz)
        pi_yz = tf.div(__pi_yz, tf.reduce_sum(__pi_yz, -1, keepdims=True))
        
        pi_yz_assign = tf.placeholder(dtype=tf.float32, shape=[cluster_num, category_size], name='pi_yz_assign')
        initialize_pi_yz = tf.assign(_pi_yz, pi_yz_assign)
        
    #================= p(h|z) =====================
    with tf.variable_scope('p_hz'):
        # mu_hz[cluster_num, embedding_size]
        # sigma_hz[cluster_num, embedding_size]
        mu_hz = tf.get_variable('mu_hz', dtype=tf.float32, initializer=tf.random_normal(shape=[cluster_num, embedding_size], mean=0, stddev=1, dtype=tf.float32))
        sigma_hz = tf.get_variable('sigma_hz', dtype=tf.float32, initializer=tf.ones([cluster_num, embedding_size], dtype=tf.float32))

        mu_hz_assign = tf.placeholder(dtype=tf.float32, shape=[cluster_num, embedding_size], name='mu_hz_assign')
        initialize_mu_hz = tf.assign(mu_hz, mu_hz_assign)
        
    #================= p(z) =====================
    with tf.variable_scope('p_z'):
        # pi_z_prior[batch_size, cluster_num]
        pi_z_prior = tf.placeholder(dtype=tf.float32, shape=[batch_size, cluster_num], name='pi_z_prior')
        _pi_z = tf.get_variable('pi_z', dtype=tf.float32, initializer=tf.ones([batch_size, cluster_num]))
        __pi_z = tf.exp(_pi_z)
        pi_z = tf.div(__pi_z, tf.reduce_sum(__pi_z, -1, keepdims=True))

        pi_z_assign = tf.placeholder(dtype=tf.float32, shape=[batch_size, cluster_num], name='pi_z_assign')
        initialize_pi_z = tf.assign(_pi_z, pi_z_assign)
    print('Decoders are constructed.')
    
#================= elbo =====================
'''
q(h|x) log p(x|h)
q(y|l) log p(l|y)
q(h|x) log q(h|x)
q(y|l) log q(y|l)
q(z|x,l)q(h|x) log p(h|z)
q(z|x,l)q(y|l) log p(y|z)
q(z|x,l) log p(z)
q(z|x,l) log q(z|x,l)
q(z|x,l)
'''
with tf.name_scope('elbo'):
    #================= q(h|x) log p(x|h) =====================
    with tf.name_scope('q_hx_log_p_xh'):
        # reduce_mean along both T and batch_size
        _tmp = tf.reshape(x, [batch_size, 1, feature_size])
        if flag_node_type == 'Bernoulli':
            elbo_q_hx_log_p_xh = tf.reduce_mean(tf.reduce_sum(_tmp*tf.log(1e-10+mu_xh) + (1.0-_tmp)*tf.log(1e-10+(1.0-mu_xh)), -1))
        elif flag_node_type == 'Gaussian':
            elbo_q_hx_log_p_xh = -0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(_tmp-mu_xh), -1))
    
    #================= q(y|l) log p(l|y) =====================
    with tf.name_scope('q_yl_log_p_ly'):
        elbo_q_yl_log_p_ly = -dls.LAA_loss_reconstr(l, pi_ly, pi_yl)
        
    #================= q(h|x) log q(h|x) =====================
    with tf.name_scope('q_hx_log_q_hx'):
        elbo_q_hx_log_q_hx = -0.5 * tf.reduce_mean(tf.reduce_sum(tf.log(1e-10+tf.square(sigma_hx)), -1))

    #================= q(y|l) log q(y|l) =====================
    with tf.name_scope('q_yl_log_q_yl'):
        elbo_q_yl_log_q_yl = tf.reduce_mean(tf.reduce_sum(pi_yl * tf.log(1e-10+pi_yl), -1))
    
    #================= q(z|x,l) =====================
    with tf.name_scope('q_zxl'):
        # p(h|z)[batch_size, T, cluster_num, 1]
        _h = tf.reshape(embedding_h, [batch_size, T, 1, embedding_size])
        _p_hz = -0.5 * tf.reduce_sum(
            tf.div(tf.square(_h-mu_hz), 1e-10+tf.square(sigma_hz)) 
            + tf.log(1e-10 + tf.square(sigma_hz)), -1, keepdims=True)
        # p_zhy[batch_size, T, cluster_num, category_size]
        _p_zhy = tf.log(1e-10+pi_yz) + _p_hz + tf.log(1e-10+tf.reshape(pi_z, [batch_size, 1, cluster_num, 1]))
        _p_zhy_max = tf.reduce_max(_p_zhy, 2, keepdims=True)
        p_zhy = tf.exp(_p_zhy - (_p_zhy_max + tf.log(1e-10+tf.reduce_sum(tf.exp(_p_zhy-_p_zhy_max), 2, keepdims=True))))
        
        # q_zxl[batch_size, cluster_num]
        # reduce_mean along both category_size and T
        _q_zxl = tf.reduce_sum(tf.reshape(pi_yl, [batch_size, 1, 1, category_size]) * p_zhy, -1)
        q_zxl = tf.reduce_mean(_q_zxl, 1)
        
        # z_index[batch_size]
        z_index = tf.argmax(q_zxl, 1)
        # cluster_pi_max[batch_size, category_size]
        # cluster_pi_avg[batch_size, category_size]
        cluster_pi_max = tf.gather(pi_yz, z_index)
        cluster_pi_avg = tf.matmul(q_zxl, pi_yz)
        
    #================= q(z|x,l)q(h|x) log p(h|z) =====================
    #================= q(h|x) log p(h|z) [batch_size, cluster_num] =====================
    with tf.name_scope('q_zxl_q_hx_log_p_hz'):
        # mu_hx[batch_size, embedding_size]
        # sigma_hx[batch_size, embedding_size]
        # mu_hz[cluster_num, embedding_size]
        # sigma_hz[cluster_num, embedding_size]
        _part_1 = tf.div(tf.square(tf.reshape(mu_hx, [batch_size, 1, embedding_size]) - mu_hz), 1e-10+tf.square(sigma_hz))
        _part_2 = tf.div(tf.square(tf.reshape(sigma_hx, [batch_size, 1, -1])), 1e-10+tf.square(sigma_hz))
        _part_3 = tf.log(1e-10 + tf.square(sigma_hz))
        # elbo_q_hx_log_p_hz[batch_size, cluster_num]
        elbo_q_hx_log_p_hz = -0.5 * tf.reduce_sum(_part_1 + _part_2 + _part_3, -1)
        elbo_q_zxl_q_hx_log_p_hz = tf.reduce_mean(tf.reduce_sum(q_zxl * elbo_q_hx_log_p_hz, -1))
    
    #================= q(z|x,l)q(y|l) log p(y|z) =====================
    #================= q(y|l) log p(y|z) [batch_size, cluster_num] =====================
    with tf.name_scope('q_zxl_q_yl_log_p_yz'):
        # pi_yz[cluster_num, category_size]
        # pi_yl[batch_size, category_size]
        # elbo_q_yl_log_p_yz[batch_size, cluster_num]
        elbo_q_yl_log_p_yz = tf.reduce_sum(tf.reshape(pi_yl, [batch_size, 1, category_size]) * tf.log(1e-10 + pi_yz), -1)
        elbo_q_zxl_q_yl_log_p_yz = tf.reduce_mean(tf.reduce_sum(q_zxl * elbo_q_yl_log_p_yz, -1))
    
    #================= q(z|x,l) log p(z) =====================
    #================= log p(z) [cluster_num] =====================
    with tf.name_scope('q_zxl_log_p_z'):
        # elbo_log_p_z[batch_size, cluster_num]
        elbo_log_p_z = tf.log(1e-10 + pi_z)
        elbo_q_zxl_log_p_z = tf.reduce_mean(tf.reduce_sum(q_zxl * elbo_log_p_z, -1))
    
    #================= q(z|x,l) log q(z|x,l) =====================
    with tf.name_scope('q_zxl_log_q_zxl'):
        # q_zxl[batch_size, cluster_num]
        elbo_q_zxl_log_q_zxl = tf.reduce_mean(tf.reduce_sum(q_zxl * tf.log(1e-10 + q_zxl), -1))
    
    #================= overall elbo =====================
    elbo = elbo_q_hx_log_p_xh + elbo_q_yl_log_p_ly - elbo_q_hx_log_q_hx - elbo_q_yl_log_q_yl         + elbo_q_zxl_q_hx_log_p_hz + elbo_q_zxl_q_yl_log_p_yz + elbo_q_zxl_log_p_z - elbo_q_zxl_log_q_zxl    
    
    q_zxl_entropy = -elbo_q_zxl_log_q_zxl
    
    with tf.variable_scope('regularization_prior'):
        mu_hz_prior_mu = tf.placeholder(dtype=tf.float32, shape=[cluster_num, embedding_size], name='mu_hz_prior_mu')
        # sigma_hz_prior_alpha = tf.placeholder(dtype=tf.float32, shape=[cluster_num, embedding_size], name='sigma_hz_prior')
        pi_yz_prior = tf.placeholder(dtype=tf.float32, shape=[cluster_num, category_size], name='pi_yz_prior')
    
        constraint_prior = 0.5*tf.reduce_mean(tf.square(mu_hz - mu_hz_prior_mu))             - tf.reduce_mean(pi_yz_prior * tf.log(1e-10+pi_yz))             - tf.reduce_mean(pi_z_prior * tf.log(1e-10+pi_z))             + tf.reduce_mean(1.0*tf.log(1e-10+tf.square(sigma_hz))+tf.div(2.0, 1e-10+tf.square(sigma_hz)))
    
    loss_overall = -elbo         + constraint_w_AE         + constraint_w_LAA         + 1.0 * constraint_prior
    
    # optimizier
    learning_rate_overall = 0.001
    optimizer_overall = tf.train.AdamOptimizer(learning_rate=learning_rate_overall).minimize(loss_overall)

    print('Clustering-based label-aware autoencoder is constructed.')

saver = tf.train.Saver()


# In[ ]:


#================= training and inference =====================
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #================= pre-train pi_yl =====================
    # assign batch variables (use whole data in one batch)
    # define majority voting regularizer
    majority_y = dls.get_majority_y(user_labels, source_num, category_size)
    # pre-train classifier
    print("Pre-train pi_yl ...")
    epochs = 50
    for epoch in range(epochs):
        _, monitor_pi_yl = sess.run([optimizer_pre_train_yl, pi_yl], 
            feed_dict={l:user_labels, pi_yl_target:majority_y})
        if epoch % 10 == 0:
            hit_num = dls.cal_hit_num(true_labels, monitor_pi_yl)
            print("epoch: {0} accuracy: {1}".format(epoch, float(hit_num)/n_samples))
    
    print("Pre-train hx_AE ...")
    epochs = 2000
    for epoch in range(epochs):
        _, monitor_loss_square_AE, monitor_mu_hx = sess.run([optimizer_AE, loss_square_AE, mu_hx], 
            feed_dict={x:feature})
        if epoch % 50 == 0:
            print("epoch: {0} loss: {1}".format(epoch, monitor_loss_square_AE))    
    
    #================= calculate initial parameters =====================
    clustering_result = KMeans(n_clusters=cluster_num).fit(np.concatenate((monitor_mu_hx, majority_y), 1))
    # pi_z_prior_cluster = np.ones([n_samples, cluster_num]) / cluster_num
    pi_z_prior_cluster = dls.convert_to_one_hot(clustering_result.labels_, cluster_num, smooth=0.2)
    _ = sess.run(initialize_mu_hz, {mu_hz_assign:clustering_result.cluster_centers_[:, 0:embedding_size]})
    # pi_yz_prior_cluster = np.ones([cluster_num, category_size]) / cluster_num
    pi_yz_prior_cluster = dls.get_cluster_majority_y(
        clustering_result.labels_, user_labels, cluster_num, source_num, category_size)
    _ = sess.run(initialize_pi_yz, {pi_yz_assign:pi_yz_prior_cluster})
    _ = sess.run(initialize_pi_z, {pi_z_assign:pi_z_prior_cluster})
    
    mu_hz_prior_mu_cluster = clustering_result.cluster_centers_[:, 0:embedding_size]
    
    predict_label = np.zeros([batch_size, category_size])
    for i in range(batch_size):
        predict_label[i] = pi_yz_prior_cluster[clustering_result.labels_[i], :]
    print("Initial clustering accuracy: {0}".format(float(dls.cal_hit_num(true_labels, predict_label)) / n_samples))
    
    #================= save current model =====================
    saved_path = saver.save(sess, './my_model')


# In[ ]:


with tf.Session() as sess:
    saver.restore(sess, './my_model')
    
    print("Train overall net ...")
    epochs = 2000
    for epoch in range(epochs):
        _, monitor_loss_overall, monitor_pi_yl, monitor_cluster_pi_max, monitor_cluster_pi_avg,             monitor_constraint_w_AE, monitor_constraint_prior = sess.run(
                [optimizer_overall, loss_overall, pi_yl, cluster_pi_max, cluster_pi_avg, constraint_w_AE, constraint_prior], 
                feed_dict={l:user_labels, x:feature, 
                           pi_z_prior:pi_z_prior_cluster, 
                           mu_hz_prior_mu:mu_hz_prior_mu_cluster, 
                           pi_yz_prior:pi_yz_prior_cluster})
        if epoch % 10 == 0:
            print("epoch: {0} loss: {1}".format(epoch, monitor_loss_overall))
            print("epoch: {0} loss: {1}".format(epoch, monitor_constraint_w_AE))
            print("epoch: {0} loss: {1}".format(epoch, monitor_constraint_prior))
            hit_num_object_level = dls.cal_hit_num(true_labels, monitor_pi_yl)
            hit_num_cluster_level_avg = dls.cal_hit_num(true_labels, monitor_cluster_pi_avg)
            print("epoch: {0} accuracy(object level): {1}".format(epoch, float(hit_num_object_level)/n_samples))
            print("epoch: {0} accuracy(cluster level avg): {1}".format(epoch, float(hit_num_cluster_level_avg)/n_samples))
    print("Training overall net. Done!")
