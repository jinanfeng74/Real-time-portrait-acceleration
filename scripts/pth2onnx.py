import os
import torch

from src.modules.warping_network import WarpingNetwork

# 参数配置字典，包含了网络的结构和其他设置
params = {
    'num_kp': 21,  # 关键点数量
    'block_expansion': 64,  # 扩展的卷积块的数量
    'max_features': 512,  # 网络中最大特征数量
    'num_down_blocks': 2,  # 网络的下采样块数
    'reshape_channel': 32,  # 重新塑造的通道数
    'estimate_occlusion_map': True,  # 是否估算遮挡图
    'dense_motion_params': {
        'block_expansion': 32,  # 变形网络的扩展块
        'max_features': 1024,  # 变形网络的最大特征
        'num_blocks': 5,  # 变形网络的块数
        'reshape_depth': 16,  # 变形网络的深度
        'compress': 4  # 变形网络的压缩因子
    }
}

# 初始化 WarpingNetwork 模型
model = WarpingNetwork(**params)

# 加载预训练模型权重
ckpt_path = "./pretrained_weights/liveportrait/base_models/warping_module.pth"
model.load_state_dict(torch.load(ckpt_path, map_location=lambda storage, loc: storage))
model.eval()  # 将模型设置为评估模式

# 获取 dense_motion_network 子模块
dense_model = model.dense_motion_network
print("Dense Model: ", dense_model)

# 构造模型输入
feature_3d = torch.randn([1, 32, 16, 64, 64], dtype=torch.float32)  # 3D特征图，随机初始化
kp_source = torch.randn([1, 21, 3], dtype=torch.float32)  # 源关键点，随机初始化
kp_drive = torch.randn([1, 21, 3], dtype=torch.float32)  # 驱动关键点，随机初始化

# 使用 dense_motion_network 进行前向推理，获得输出
outputs = dense_model(feature_3d, kp_source, kp_drive)

# 打印输出张量的形状
for k, v in outputs.items():
    print(k, v.shape)

# 将 dense_motion_network 导出为 ONNX 格式
torch.onnx.export(
    dense_model,
    (feature_3d, kp_source, kp_drive),
    os.path.join("./script/warping_module.onxx"),  # 导出的文件路径
    export_params=True,  # 导出模型参数
    opset_version=20,  # 指定 ONNX opset 版本
    do_constant_folding=True,  # 开启常量折叠优化
    input_names=['feature_3d', 'kp_driving', 'kp_source'],  # 输入张量的名称
    output_names=['mask', 'deformation', 'occlusion_map'],  # 输出张量的名称
    dynamic_axes={
        'feature_3d': {0: 'batch_size'},  # 允许批量大小变化
        'kp_source': {0: 'batch_size'},  # 允许批量大小变化
        'kp_driving': {0: 'batch_size'},  # 允许批量大小变化
        'out': {0: 'batch_size'},  # 允许批量大小变化
    }
)

# 对整个 WarpingNetwork 模型进行前向推理
outputs = model(feature_3d, kp_source, kp_drive)

# 打印输出张量的形状
for k, v in outputs.items():
    print(k, v.shape)

# 将整个 WarpingNetwork 模型导出为 ONNX 格式
torch.onnx.export(
    model,
    (feature_3d, kp_source, kp_drive),
    os.path.join("./warping_all.onnx"),  # 导出的文件路径
    export_params=True,  # 导出模型参数
    opset_version=20,  # 指定 ONNX opset 版本
    do_constant_folding=True,  # 开启常量折叠优化
    input_names=['feature_3d', 'kp_driving', 'kp_source'],  # 输入张量的名称
    output_names=['occlusion_map', 'deformation', 'out'],  # 输出张量的名称
    dynamic_axes={
        'feature_3d': {0: 'batch_size'},  # 允许批量大小变化
        'kp_source': {0: 'batch_size'},  # 允许批量大小变化
        'kp_driving': {0: 'batch_size'},  # 允许批量大小变化
        'out': {0: 'batch_size'},  # 允许批量大小变化
    }
)

print("ONNX export completed successfully.")
