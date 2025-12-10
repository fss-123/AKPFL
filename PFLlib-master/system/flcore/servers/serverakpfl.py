# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import random
import time
from flcore.clients.clientakpfl import clientRep
from flcore.servers.serverbase import Server
import torch


class FedRep(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # 初始化全局模型
        self.model = self.initialize_global_model()

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientRep)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []

    def initialize_global_model(self):
        """ 初始化全局模型，与客户端的全局模型一致 """
        model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # conv1
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 64, kernel_size=3),  # conv2
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 6 * 6, 512),  # 调整输出为 512 维
            torch.nn.ReLU()
        )
        return model

    def rename_state_dict_keys(self, state_dict):
        """重命名服务器端的 state_dict 键，以匹配客户端的命名"""
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('0'):
                new_key = key.replace('0', 'conv1')
            elif key.startswith('3'):
                new_key = key.replace('3', 'conv2')
            elif key.startswith('7'):
                new_key = key.replace('7', 'fc1')
            else:
                new_key = key
            new_state_dict[new_key] = value
        return new_state_dict

    def clean_nan_inf_values(self, tensor):
        """清除或替换张量中的 NaN 和 Inf 值"""
        tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)  # 替换 NaN 为 0
        tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)  # 替换 inf 为 0
        return tensor

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])
            # 在每轮训练结束后将聚合后的全局模型发送给客户端
            self.send_models()

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientRep)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                               client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_weights.append(client.train_samples)
                cleaned_model = {k: self.clean_nan_inf_values(v) for k, v in client.model.base.state_dict().items()}
                client.model.base.load_state_dict(cleaned_model)  # 加载清理过的模型
                if self.is_model_valid(client.model.base):
                    # 调用新增的模型检测方法
                    self.uploaded_models.append(client.model.base)
                else:
                    print(f"Client {client.id} uploaded an invalid model, skipping.")

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    # 新增：检查模型中是否包含 NaN 或 Inf
    def is_model_valid(self, model):
        for param in model.parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                return False
        return True







