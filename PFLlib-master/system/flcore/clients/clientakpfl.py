import torch
import time
import numpy as np
from flcore.clients.clientbase import Client


class clientRep(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        # 初始化父类
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # 选择设备，使用 GPU (cuda) 或 CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 将模型移动到设备 (cuda 或 CPU)
        self.model = self.model.to(self.device)

        # 初始化全局模型的优化器，使用 SGD 优化器，设定较小的学习率 (如 1e-2)，并加入 L2 正则化以防止过拟合
        self.optimizer = torch.optim.SGD(self.model.base.parameters(), lr=1e-2, weight_decay=1e-5, momentum=0.9)
        self.optimizer_per = torch.optim.SGD(self.model.head.parameters(), lr=1e-2, weight_decay=1e-5, momentum=0.9)

        # 设置学习率调度器，采用指数衰减策略
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma  # 学习率衰减系数
        )
        self.learning_rate_scheduler_per = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_per,
            gamma=args.learning_rate_decay_gamma  # 学习率衰减系数
        )

        # 训练的超参数，包括局部训练轮次和全局训练轮次
        self.plocal_epochs = args.plocal_epochs
        self.local_epochs = args.local_epochs  # 加入全局模型的本地训练轮次
        self.alpha = args.alpha  # 动态平滑系数的初始值
        self.meta_learning_rate = 0.01  # MAML元学习率

    # 动态调整学习率函数
    def adjust_learning_rate(self, optimizer, base_lr, decay_rate, round_num, total_rounds, min_lr=1e-5):
        """
        动态调整学习率函数，基于指数衰减或退火策略。
        :param optimizer: 优化器
        :param base_lr: 初始学习率
        :param decay_rate: 学习率衰减因子
        :param round_num: 当前轮次
        :param total_rounds: 总训练轮次
        :param min_lr: 最小学习率
        """
        # 计算新的学习率，基于指数衰减
        new_lr = base_lr * (decay_rate ** (round_num / total_rounds))
        # 确保学习率不低于设定的最小值
        new_lr = max(new_lr, min_lr)

        # 更新优化器的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    # 定义一个函数用于计算两个模型参数的加权平均（模型融合）
    def model_fusion(self, model1_params, model2_params, fusion_weight, kernel_type="matern", sigma=1.0):

        # 计算当前模型和上一轮模型之间的核距离
        kernel_distance = self.calculate_kernel_distance(model1_params, model2_params, kernel_type, sigma)

        # 根据核距离动态调整权重
        adjusted_weight = fusion_weight * torch.exp(-kernel_distance)  # 根据核距离调整权重
        adjusted_weight = max(0.05, min(adjusted_weight.item(), 0.95))  # 限制权重范围在 [0.05, 0.95]

        fused_params = []
        for param1, param2 in zip(model1_params, model2_params):
            fused_param = fusion_weight * param1 + (1 - adjusted_weight) * param2  # 加权平均融合模型参数
            fused_params.append(fused_param.clone())
        return fused_params

    # 基于损失的个性化加权策略动态调整融合权
    def adaptive_fusion_weight(self, prev_loss, current_loss):
        loss_diff = prev_loss - current_loss
        if loss_diff > 0.01:  # 损失下降显著，个性化效果好
            return 0.3  # 更重视个性化部分
        elif loss_diff < -0.01:  # 损失上升，增加全局模型权重
            return 0.7  # 更重视全局模型部分
        else:
            return 0.5  # 保持平衡

    # 定义基于本地特征的正则化
    def personalized_regularization_loss(self):
        reg_loss = 0
        for param in self.model.head.parameters():
            reg_loss += torch.norm(param, p=2)  # L2 正则化
        return reg_loss * 0.01  # 设置正则化权重

    # 定义线性核函数计算两个向量的内积
    def linear_kernel(self, x, y):
        return torch.dot(x, y)

    # 定义 RBF 核函数，用于计算两个向量的高斯核距离
    def rbf_kernel(self, x, y, sigma=1.0):
        diff = x - y
        return torch.exp(-torch.norm(diff) ** 2 / (2 * (sigma ** 2)))

    # 定义多项式核函数
    def polynomial_kernel(self, x, y, d=3, gamma=1.0, c=1.0):
        return (gamma * torch.dot(x, y) + c) ** d

    # 定义拉普拉斯核函数
    def laplacian_kernel(self, x, y, sigma=1.0):
        diff = torch.abs(x - y)
        return torch.exp(-torch.sum(diff) / (2 * (sigma ** 2)))

    # 定义 Matern 核函数，用于计算两个向量之间的距离
    def matern_kernel(self, x, y, nu=1.5, length_scale=1.0):
        """
        Matern 核函数，可以通过参数 nu 调整平滑度。对于 nu=1.5 或 2.5 分别对应不同平滑度。
        """
        distance = torch.norm(x - y)
        if nu == 0.5:
            return torch.exp(-distance / length_scale)
        elif nu == 1.5:
            factor = (1.0 + (np.sqrt(3) * distance / length_scale))
            return factor * torch.exp(-np.sqrt(3) * distance / length_scale)
        elif nu == 2.5:
            factor = (1.0 + (np.sqrt(5) * distance / length_scale) + (5.0 * distance ** 2) / (3.0 * length_scale ** 2))
            return factor * torch.exp(-np.sqrt(5) * distance / length_scale)
        else:
            raise ValueError("Unsupported value of nu. Choose 0.5, 1.5, or 2.5.")

    # # 使用核函数计算两个模型参数集的距离
    # def calculate_kernel_distance(self, h_current, h_previous, kernel_type="adaptive", sigma=1.0):

    #     # 将模型参数展平为一维向量，便于距离计算
    #     h_current_flat = torch.cat([param.view(-1).to(self.device) for param in h_current])
    #     h_previous_flat = torch.cat([param.view(-1).to(self.device) for param in h_previous])
    #     #如果指定为自适应选择，则测试多个核函数的表现
    #     if kernel_type == "adaptive":
    #         kernel_scores = {}
    #         # 遍历不同核函数，计算相似性（可扩展更多核函数类型）
    #         kernel_scores["linear"] = self.linear_kernel(h_current_flat, h_previous_flat)
    #         kernel_scores["rbf"] = self.rbf_kernel(h_current_flat, h_previous_flat, sigma)
    #         kernel_scores["polynomial"] = self.polynomial_kernel(h_current_flat, h_previous_flat)
    #         kernel_scores["laplacian"] = self.laplacian_kernel(h_current_flat, h_previous_flat, sigma)
    #         kernel_scores["matern"] = self.matern_kernel(h_current_flat, h_previous_flat, nu=1.5)
    #         # 找到最优核函数类型（相似性越高越好）
    #         best_kernel_type = max(kernel_scores, key=kernel_scores.get)
    #         # 输出选择的核函数类型
    #         #print(f"Selected kernel type: {best_kernel_type}")
    #         # 使用最佳核函数计算最终的核距离
    #         if best_kernel_type == "linear":
    #             k_xx = self.linear_kernel(h_current_flat, h_current_flat)
    #             k_yy = self.linear_kernel(h_previous_flat, h_previous_flat)
    #             k_xy = self.linear_kernel(h_current_flat, h_previous_flat)
    #         elif best_kernel_type == "rbf":
    #             k_xx = self.rbf_kernel(h_current_flat, h_current_flat, sigma)
    #             k_yy = self.rbf_kernel(h_previous_flat, h_previous_flat, sigma)
    #             k_xy = self.rbf_kernel(h_current_flat, h_previous_flat, sigma)
    #         elif best_kernel_type == "polynomial":
    #             k_xx = self.polynomial_kernel(h_current_flat, h_current_flat)
    #             k_yy = self.polynomial_kernel(h_previous_flat, h_previous_flat)
    #             k_xy = self.polynomial_kernel(h_current_flat, h_previous_flat)
    #         elif best_kernel_type == "laplacian":
    #             k_xx = self.laplacian_kernel(h_current_flat, h_current_flat, sigma)
    #             k_yy = self.laplacian_kernel(h_previous_flat, h_previous_flat, sigma)
    #             k_xy = self.laplacian_kernel(h_current_flat, h_previous_flat, sigma)
    #         elif best_kernel_type == "matern":
    #             k_xx = self.matern_kernel(h_current_flat, h_current_flat, nu=1.5)
    #             k_yy = self.matern_kernel(h_previous_flat, h_previous_flat, nu=1.5)
    #             k_xy = self.matern_kernel(h_current_flat, h_previous_flat, nu=1.5)
    #         else:
    #             raise  ValueError(f"Unsupported kernel type: {best_kernel_type}")
    #          # 根据公式计算核距离
    #         kernel_distance = torch.sqrt(k_xx + k_yy - 2 * k_xy)
    #         return kernel_distance
    #     else:
    #         # 根据指定核函数类型计算距离
    #         if kernel_type == "linear":
    #             k_xx = self.linear_kernel(h_current_flat, h_current_flat)
    #             k_yy = self.linear_kernel(h_previous_flat, h_previous_flat)
    #             k_xy = self.linear_kernel(h_current_flat, h_previous_flat)
    #         elif kernel_type == "rbf":
    #             k_xx = self.rbf_kernel(h_current_flat, h_current_flat, sigma)
    #             k_yy = self.rbf_kernel(h_previous_flat, h_previous_flat, sigma)
    #             k_xy = self.rbf_kernel(h_current_flat, h_previous_flat, sigma)
    #         elif kernel_type == "polynomial":
    #             k_xx = self.polynomial_kernel(h_current_flat, h_current_flat)
    #             k_yy = self.polynomial_kernel(h_previous_flat, h_previous_flat)
    #             k_xy = self.polynomial_kernel(h_current_flat, h_previous_flat)
    #         elif kernel_type == "laplacian":
    #             k_xx = self.laplacian_kernel(h_current_flat, h_current_flat, sigma)
    #             k_yy = self.laplacian_kernel(h_previous_flat, h_previous_flat, sigma)
    #             k_xy = self.laplacian_kernel(h_current_flat, h_previous_flat, sigma)
    #         elif kernel_type == "matern":
    #             k_xx = self.matern_kernel(h_current_flat, h_current_flat, nu=1.5)
    #             k_yy = self.matern_kernel(h_previous_flat, h_previous_flat, nu=1.5)
    #             k_xy = self.matern_kernel(h_current_flat, h_previous_flat, nu=1.5)
    #         else:
    #             raise ValueError("Unsupported kernel type. Choose 'linear', 'rbf', 'polynomial', or 'laplacian'.")
    #         kernel_distance = torch.sqrt(k_xx + k_yy - 2 * k_xy)
    #         return kernel_distance

    def calculate_kernel_distance(self, h_current, h_previous, kernel_type="adaptive", sigma=1.0):

        # 将模型参数展平为一维向量，便于距离计算
        h_current_flat = torch.cat([param.view(-1).to(self.device) for param in h_current])
        h_previous_flat = torch.cat([param.view(-1).to(self.device) for param in h_previous])

        if kernel_type == "adaptive":

            # 存储各核函数计算的距离
            kernel_distances = {}

            # 计算线性核距离
            k_xx_linear = self.linear_kernel(h_current_flat, h_current_flat)
            k_yy_linear = self.linear_kernel(h_previous_flat, h_previous_flat)
            k_xy_linear = self.linear_kernel(h_current_flat, h_previous_flat)
            kernel_distances["linear"] = torch.sqrt(k_xx_linear + k_yy_linear - 2 * k_xy_linear)

            # 计算 RBF 核距离
            k_xx_rbf = self.rbf_kernel(h_current_flat, h_current_flat, sigma)
            k_yy_rbf = self.rbf_kernel(h_previous_flat, h_previous_flat, sigma)
            k_xy_rbf = self.rbf_kernel(h_current_flat, h_previous_flat, sigma)
            kernel_distances["rbf"] = torch.sqrt(k_xx_rbf + k_yy_rbf - 2 * k_xy_rbf)

            # 计算多项式核距离
            k_xx_poly = self.polynomial_kernel(h_current_flat, h_current_flat)
            k_yy_poly = self.polynomial_kernel(h_previous_flat, h_previous_flat)
            k_xy_poly = self.polynomial_kernel(h_current_flat, h_previous_flat)
            kernel_distances["polynomial"] = torch.sqrt(k_xx_poly + k_yy_poly - 2 * k_xy_poly)

            # 计算拉普拉斯核距离
            k_xx_laplacian = self.laplacian_kernel(h_current_flat, h_current_flat, sigma)
            k_yy_laplacian = self.laplacian_kernel(h_previous_flat, h_previous_flat, sigma)
            k_xy_laplacian = self.laplacian_kernel(h_current_flat, h_previous_flat, sigma)
            kernel_distances["laplacian"] = torch.sqrt(k_xx_laplacian + k_yy_laplacian - 2 * k_xy_laplacian)

            # 计算 Matern 核距离
            k_xx_matern = self.matern_kernel(h_current_flat, h_current_flat, nu=1.5)
            k_yy_matern = self.matern_kernel(h_previous_flat, h_previous_flat, nu=1.5)
            k_xy_matern = self.matern_kernel(h_current_flat, h_previous_flat, nu=1.5)
            kernel_distances["matern"] = torch.sqrt(k_xx_matern + k_yy_matern - 2 * k_xy_matern)

            # 找到最优核函数类型（相似性越高越好，即距离最小）
            best_kernel_type = min(kernel_distances, key=kernel_distances.get)

            # 输出选择的核函数类型
            # print(f"Selected kernel type: {best_kernel_type}")

            # 返回最小的核距离
            return kernel_distances[best_kernel_type]

        else:
            # 使用指定核函数计算核距离
            if kernel_type == "linear":
                k_xx = self.linear_kernel(h_current_flat, h_current_flat)
                k_yy = self.linear_kernel(h_previous_flat, h_previous_flat)
                k_xy = self.linear_kernel(h_current_flat, h_previous_flat)
            elif kernel_type == "rbf":
                k_xx = self.rbf_kernel(h_current_flat, h_current_flat, sigma)
                k_yy = self.rbf_kernel(h_previous_flat, h_previous_flat, sigma)
                k_xy = self.rbf_kernel(h_current_flat, h_previous_flat, sigma)
            elif kernel_type == "polynomial":
                k_xx = self.polynomial_kernel(h_current_flat, h_current_flat)
                k_yy = self.polynomial_kernel(h_previous_flat, h_previous_flat)
                k_xy = self.polynomial_kernel(h_current_flat, h_previous_flat)
            elif kernel_type == "laplacian":
                k_xx = self.laplacian_kernel(h_current_flat, h_current_flat, sigma)
                k_yy = self.laplacian_kernel(h_previous_flat, h_previous_flat, sigma)
                k_xy = self.laplacian_kernel(h_current_flat, h_previous_flat, sigma)
            elif kernel_type == "matern":
                k_xx = self.matern_kernel(h_current_flat, h_current_flat, nu=1.5)
                k_yy = self.matern_kernel(h_previous_flat, h_previous_flat, nu=1.5)
                k_xy = self.matern_kernel(h_current_flat, h_previous_flat, nu=1.5)
            else:
                raise ValueError("Unsupported kernel type. Choose 'linear', 'rbf', 'polynomial', or 'laplacian'.")

            # 根据公式计算核距离
            kernel_distance = torch.sqrt(k_xx + k_yy - 2 * k_xy)
            return kernel_distance

    # 动态调整平滑系数公式
    def dynamic_alpha(self, kernel_distance, q_factor):
        return kernel_distance * q_factor

    # 定义个性化头部的 Q 函数，随轮次递减，Q(t) = 1 - s / τ_h
    def Q_function_head(self):
        return 1 - self.current_epoch_head / self.plocal_epochs

    # 定义全局模型的 Q 函数，随轮次递减，Q(t) = 1 - s / τ_h
    def Q_function_global(self):
        return 1 - self.current_epoch_global / self.local_epochs

    # MAML元学习过程
    def maml_inner_loop(self, x, y):
        # 获取模型的初始参数
        original_params = [p.clone() for p in self.model.parameters()]

        # 前向传播，计算损失
        output = self.model(x)
        loss = self.loss(output, y)

        # 反向传播并计算梯度
        self.optimizer.zero_grad()
        loss.backward()

        # 使用元学习率更新模型参数
        for param in self.model.parameters():
            param.data -= self.meta_learning_rate * param.grad

        # 返回更新后的模型参数
        updated_params = [p.clone() for p in self.model.parameters()]

        # 恢复模型的初始参数（用于后续客户端训练）
        for original, current in zip(original_params, self.model.parameters()):
            current.data = original.data.clone()

        return updated_params

    # 训练过程，包括个性化头部和全局模型的更新，结合MAML方法
    def train(self):
        trainloader = self.load_train_data()  # 加载本地训练数据
        start_time = time.time()  # 记录训练开始时间

        # 初始化损失值，prev_loss设置为较大值，current_loss设置为零开始
        prev_loss = float('inf')
        current_loss = 0.0

        # 动态调整学习率相关参数
        base_lr = 1e-2  # 默认初始学习率
        decay_rate = 0.9  # 默认指数衰减因子
        min_lr = 1e-5  # 设置最小学习率，避

        # 1. 使用 MAML 进行个性化元训练
        self.model.train()  # 将模型设置为训练模式

        for i, (x, y) in enumerate(trainloader):
            x = x.to(self.device)
            y = y.to(self.device)
            # 使用 MAML 内循环更新参数
            updated_params = self.maml_inner_loop(x, y)

            # 将更新后的参数加载回模型
            for param, updated_param in zip(self.model.parameters(), updated_params):
                param.data = updated_param.data.clone()

        # 2. 训练个性化头部模型
        # 冻结全局模型的参数，仅训练个性化头部
        for param in self.model.base.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True

        # 保存上一轮次的个性化头部参数
        h_t_minus_1 = [p.data.clone().to(self.device) for p in self.model.head.parameters()]

        for epoch in range(self.plocal_epochs // 2):  # 减少个性化训练轮次
            self.current_epoch_head = epoch
            # 动态调整个性化优化器的学习率
            self.adjust_learning_rate(self.optimizer_per, base_lr, decay_rate, self.current_epoch_head,
                                      self.local_epochs, min_lr)
            for i, (x, y) in enumerate(trainloader):
                x = x.to(self.device)
                y = y.to(self.device)

                # 前向传播，计算损失
                output = self.model(x)
                loss = self.loss(output, y)

                # 增加基于本地特征的正则化项
                total_loss = loss + self.personalized_regularization_loss()

                # 反向传播，更新个性化头部参数 h_t^s
                self.optimizer_per.zero_grad()
                total_loss.backward()

                # 在反向传播计算完梯度后加入梯度裁剪，防止梯度过大（爆炸）
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer_per.step()

                # 计算核距离并应用平滑更新公式
                with torch.no_grad():
                    h_t = [p.data.clone().to(self.device) for p in self.model.head.parameters()]
                    kernel_distance = self.calculate_kernel_distance(h_t, h_t_minus_1,
                                                                     kernel_type="adaptive")  # 使用RBF核函数
                    q_factor = self.Q_function_head()
                    self.alpha = self.dynamic_alpha(kernel_distance, q_factor)
                    # 使用动态区间，根据当前训练的轮次逐渐扩大 alpha 的取值范围
                    min_alpha = 0.05 + (self.current_epoch_global / self.local_epochs) * 0.4
                    max_alpha = 0.9 - (self.current_epoch_global / self.local_epochs) * 0.4
                    self.alpha = max(min_alpha, min(self.alpha, max_alpha))

                    for ht, ht_1 in zip(self.model.head.parameters(), h_t_minus_1):
                        ht.data = self.alpha * ht_1 + (1 - self.alpha) * ht.data

                # 更新上一轮次的个性化头部参数
                h_t_minus_1 = h_t

        # 3. 训练全局模型部分
        max_local_epochs = self.local_epochs * 2  # 增加全局模型训练轮次
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        # 解冻全局模型参数，冻结个性化头部模型的参数
        for param in self.model.base.parameters():
            param.requires_grad = True
        for param in self.model.head.parameters():
            param.requires_grad = False

        # 保存上一轮次的全局模型参数
        phi_t_minus_1 = [p.data.clone().to(self.device) for p in self.model.base.parameters()]

        for epoch in range(max_local_epochs):  # 加强全局模型的参与度
            self.current_epoch_global = epoch
            # 动态调整全局优化器的学习率

            self.adjust_learning_rate(self.optimizer, base_lr, decay_rate, self.current_epoch_global, self.local_epochs,
                                      min_lr)

            for i, (x, y) in enumerate(trainloader):
                x = x.to(self.device)
                y = y.to(self.device)

                # 前向传播，计算损失
                output = self.model(x)
                loss = self.loss(output, y)

                # 反向传播，更新全局模型参数
                self.optimizer.zero_grad()
                loss.backward()

                # 在反向传播计算完梯度后加入梯度裁剪，防止梯度过大（爆炸）
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                # 核距离与全局模型的平滑更新
                with torch.no_grad():
                    phi_t = [p.data.clone().to(self.device) for p in self.model.base.parameters()]
                    kernel_distance = self.calculate_kernel_distance(phi_t, phi_t_minus_1,
                                                                     kernel_type="adaptive")  # 使用RBF核函数
                    q_factor = self.Q_function_global()
                    self.alpha = self.dynamic_alpha(kernel_distance, q_factor)

                    # 使用动态区间，根据当前训练的轮次逐渐扩大 alpha 的取值范围
                    min_alpha = 0.05 + (self.current_epoch_global / self.local_epochs) * 0.4
                    max_alpha = 0.9 - (self.current_epoch_global / self.local_epochs) * 0.4
                    self.alpha = max(min_alpha, min(self.alpha, max_alpha))

                    for phi, phi_1 in zip(self.model.base.parameters(), phi_t_minus_1):
                        phi.data = self.alpha * phi_1 + (1 - self.alpha) * phi.data

                # 更新上一轮次的全局模型参数
                phi_t_minus_1 = phi_t

        # 增加的模型融合策略：将全局模型和个性化模型的参数进行融合
        # 根据损失变化动态调整融合权重
        fusion_weight = self.adaptive_fusion_weight(prev_loss, current_loss)  # 融合权重，可以根据性能动态调整
        with torch.no_grad():
            head_params = [p.data.clone().to(self.device) for p in self.model.head.parameters()]  # 获取个性化头部参数
            base_params = [p.data.clone().to(self.device) for p in self.model.base.parameters()]  # 获取全局基础参数

            # 应用模型融合，将全局和个性化参数融合
            fused_head_params = self.model_fusion(head_params, h_t_minus_1, fusion_weight)  # 融合个性化头部参数
            fused_base_params = self.model_fusion(base_params, phi_t_minus_1, fusion_weight)  # 融合全局基础参数

            # 用融合后的参数更新模型参数
            for param, fused_param in zip(self.model.head.parameters(), fused_head_params):
                param.data = fused_param.clone()  # 更新个性化头部参数
            for param, fused_param in zip(self.model.base.parameters(), fused_base_params):
                param.data = fused_param.clone()  # 更新全局基础参数

        # 如果启用了学习率衰减，更新学习率
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
            self.learning_rate_scheduler_per.step()

        # 记录训练时间
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    # 从全局模型加载参数并更新
    def set_parameters(self, base):
        for new_param, old_param in zip(base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()

