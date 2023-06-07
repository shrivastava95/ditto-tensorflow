import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf
import copy

from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad, cosine_sim, softmax, norm_grad, l2_clip, get_stdev
from flearn.utils.model_utils import batch_data, gen_batch, gen_epoch, gen_batch_celeba
from flearn.utils.language_utils import letter_to_vec, word_to_indices


def process_x(raw_x_batch):
    x_batch = [word_to_indices(word) for word in raw_x_batch]
    x_batch = np.array(x_batch)
    return x_batch

def process_y(raw_y_batch):
    y_batch = [letter_to_vec(c) for c in raw_y_batch]
    return y_batch


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using global-regularized multi-task learning to Train')
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)

    def train(self):
        print('---{} workers per communication round---'.format(self.clients_per_round))

        np.random.seed(1234567+self.seed)
        corrupt_id = np.random.choice(range(len(self.clients)), size=self.num_corrupted, replace=False)
        print(corrupt_id)

        if self.dataset == 'shakespeare':
            for c in self.clients:
                c.train_data['y'], c.train_data['x'] = process_y(c.train_data['y']), process_x(c.train_data['x'])
                c.test_data['y'], c.test_data['x'] = process_y(c.test_data['y']), process_x(c.test_data['x'])

        batches = {}
        for idx, c in enumerate(self.clients):
            if idx in corrupt_id:
                c.train_data['y'] = np.asarray(c.train_data['y'])
                if self.dataset == 'celeba':
                    c.train_data['y'] = 1 - c.train_data['y']
                elif self.dataset == 'femnist':
                    c.train_data['y'] = np.random.randint(0, 62, len(c.train_data['y']))  # [0, 62)
                elif self.dataset == 'shakespeare':
                    c.train_data['y'] = np.random.randint(0, 80, len(c.train_data['y']))
                elif self.dataset == "vehicle":
                    c.train_data['y'] = c.train_data['y'] * -1
                elif self.dataset == "fmnist":
                    c.train_data['y'] = np.random.randint(0, 10, len(c.train_data['y']))

            if self.dataset == 'celeba':
                # due to a different data storage format
                batches[c] = gen_batch_celeba(c.train_data, self.batch_size, self.num_rounds * self.local_iters)
            else:
                batches[c] = gen_batch(c.train_data, self.batch_size, self.num_rounds * self.local_iters)


        for i in range(self.num_rounds + 1):
            if i % self.eval_every == 0 and i > 0:
                tmp_models = []
                for idx in range(len(self.clients)):
                    tmp_models.append(self.local_models[idx])

                num_train, num_correct_train, loss_vector = self.train_error(tmp_models)
                avg_train_loss = np.dot(loss_vector, num_train) / np.sum(num_train)
                num_test, num_correct_test, _ = self.test(tmp_models)
                tqdm.write('At round {} training accu: {}, loss: {}'.format(i, np.sum(num_correct_train) * 1.0 / np.sum(num_train), avg_train_loss))
                tqdm.write('At round {} test accu: {}'.format(i, np.sum(num_correct_test) * 1.0 / np.sum(num_test)))
                non_corrupt_id = np.setdiff1d(range(len(self.clients)), corrupt_id)
                tqdm.write('At round {} malicious test accu: {}'.format(i, np.sum(num_correct_test[corrupt_id]) * 1.0 / np.sum(num_test[corrupt_id])))
                tqdm.write('At round {} benign test accu: {}'.format(i, np.sum(num_correct_test[non_corrupt_id]) * 1.0 / np.sum(num_test[non_corrupt_id])))
                print("variance of the performance: ", np.var(num_correct_test[non_corrupt_id] / num_test[non_corrupt_id]))


            # weighted sampling
            indices, selected_clients = self.select_clients(round=i, num_clients=self.clients_per_round)

            csolns = []
            losses = []

            for idx in indices:
                w_global_idx = copy.deepcopy(self.global_model)
                c = self.clients[idx]
                for _ in range(self.local_iters):
                    data_batch = next(batches[c])

                    # local
                    self.client_model.set_params(self.local_models[idx])
                    _, grads, _ = c.solve_sgd(data_batch)  


                    if self.dynamic_lam:

                        model_tmp = copy.deepcopy(self.local_models[idx])
                        model_best = copy.deepcopy(self.local_models[idx])
                        tmp_loss = 10000
                        # pick a lambda locally based on validation data
                        for lam_id, candidate_lam in enumerate([0.1, 1, 2]):
                            for layer in range(len(grads[1])):
                                eff_grad = grads[1][layer] + candidate_lam * (self.local_models[idx][layer] - self.global_model[layer])
                                model_tmp[layer] = self.local_models[idx][layer] - self.learning_rate * eff_grad

                            c.set_params(model_tmp)
                            l = c.get_val_loss()
                            if l < tmp_loss:
                                tmp_loss = l
                                model_best = copy.deepcopy(model_tmp)

                        self.local_models[idx] = copy.deepcopy(model_best)

                    else:

                        for layer in range(len(grads[1])):
                            # ditto original local update scheme - local objective + consistency loss between weights
                            eff_grad = grads[1][layer] + self.lam * (self.local_models[idx][layer] - self.global_model[layer])
                            self.local_models[idx][layer] = self.local_models[idx][layer] - self.learning_rate * eff_grad

                            # ditto modified local update scheme - local objective + KD loss between logits

                    # global
                    self.client_model.set_params(w_global_idx)
                    loss = c.get_loss() 
                    losses.append(loss)
                    _, grads, _ = c.solve_sgd(data_batch)
                    w_global_idx = self.client_model.get_params()


                # get the difference (global model updates)
                diff = [u - v for (u, v) in zip(w_global_idx, self.global_model)]


                # send the malicious updates
                if idx in corrupt_id:
                    if self.boosting:
                        # scale malicious updates
                        diff = [self.clients_per_round * u for u in diff]
                    elif self.random_updates:
                        # send random updates
                        stdev_ = get_stdev(diff)
                        diff = [np.random.normal(0, stdev_, size=u.shape) for u in diff]

                if self.q == 0:
                    csolns.append(diff)
                else:
                    csolns.append((np.exp(self.q * loss), diff))

            if self.q != 0:
                avg_updates = self.aggregate(csolns)
            else:
                if self.gradient_clipping:
                    csolns = l2_clip(csolns)

                expected_num_mali = int(self.clients_per_round * self.num_corrupted / len(self.clients))

                if self.median:
                    avg_updates = self.median_average(csolns)
                elif self.k_norm:
                    avg_updates = self.k_norm_average(self.clients_per_round - expected_num_mali, csolns)
                elif self.krum:
                    avg_updates = self.krum_average(self.clients_per_round - expected_num_mali - 2, csolns)
                elif self.mkrum:
                    m = self.clients_per_round - expected_num_mali
                    avg_updates = self.mkrum_average(self.clients_per_round - expected_num_mali - 2, m, csolns)
                else:
                    avg_updates = self.simple_average(csolns)

            # update the global model
            for layer in range(len(avg_updates)):
                self.global_model[layer] += avg_updates[layer]

        # #### Ishaan: Added this section in order to finetune.
        # # local finetuning
        # init_model = copy.deepcopy(self.latest_model)

        # after_test_accu = []
        # test_samples = []
        # for idx, c in enumerate(self.clients):
        #     c.set_params(init_model)
        #     local_model = copy.deepcopy(init_model)
        #     for _ in range(max(int(self.finetune_iters * c.train_samples / self.batch_size), self.finetune_iters)):
        #         c.set_params(local_model)
        #         data_batch = next(batches[c])
        #         _, grads, _ = c.solve_sgd(data_batch)
        #         for j in range(len(grads[1])):
        #             eff_grad = grads[1][j] + self.lam * (local_model[j] - init_model[j])
        #             local_model[j] = local_model[j] - self.learning_rate * self.decay_factor * eff_grad
        #     c.set_params(local_model)
        #     tc, _, num_test = c.test()
        #     after_test_accu.append(tc)
        #     test_samples.append(num_test)

        # after_test_accu = np.asarray(after_test_accu)
        # test_samples = np.asarray(test_samples)
        # tqdm.write('final test accu: {}'.format(np.sum(after_test_accu) * 1.0 / np.sum(test_samples)))
        # tqdm.write('final malicious test accu: {}'.format(np.sum(
        #     after_test_accu[corrupt_id]) * 1.0 / np.sum(test_samples[corrupt_id])))
        # tqdm.write('final benign test accu: {}'.format(np.sum(
        #     after_test_accu[non_corrupt_id]) * 1.0 / np.sum(test_samples[non_corrupt_id])))
        # print("variance of the performance: ",
        #       np.var(after_test_accu[non_corrupt_id] / test_samples[non_corrupt_id]))
        # ###
