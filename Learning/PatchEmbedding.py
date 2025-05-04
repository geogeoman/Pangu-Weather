class PatchEmbedding: 
  # 隐式调用PanguModel: output, output_surface = model(input, input_surface)
  # 将参数输入前向过程: x = self._input_layer(input, input_surface), 此处的输入传输至forward函数
  # 调用代码返回 x: self._input_layer = PatchEmbedding((2, 4, 4), 192)，此处的输入传输至__init__(构造)函数
  def __init__(self, patch_size, dim):
    '''
    Patch embedding operation
    We then performed patch embedding to reduce the spatial resolution and combined the down-sampled data into a 3D cube.
    '''
    # Here we use convolution to partition data into cubes
    # Convolution卷积，3D卷积核，patch_size = (2, 4, 4) as in the original paper， dim = 192
    self.conv = Conv3d(input_dims=5, output_dims=dim, kernel_size=patch_size, stride=patch_size)
    """
    1.input_dims=5: 对应输入数据的5个气象变量通道(如温度、湿度、风速等)，每个通道代表不同气压层的观测数据
    2.output_dims=dim (192): 输出特征维度, 这是Pangu-Weather论文中设定的隐藏层大小, 通过卷积将5维输入投影到192维特征空间
    3.kernel_size=patch_size (2,4,4) 三维卷积核尺寸：(垂直方向, 纬度方向, 经度方向)
      2: 垂直方向覆盖2个气压层
      4x4: 水平方向4°x4°的空间范围(对应地球约440km x 440km区域)
    4.stride=patch_size (2,4,4) 与kernel_size相同的步长设置,实现非重叠的patch划分:
      垂直方向: 每2层作为一个处理单元
      水平方向: 将地球网格划分为不重叠的4°x4°区块

    技术特点：
    1.这是一个等步长卷积(stride=kernel_size),直接将输入数据划分为空间块(patch)
    2.替代了传统Vision Transformer中显式的patch划分+线性投影，通过卷积同时完成：
      空间局部特征提取
      通道维度变换(5→192)
    3.特别适合处理地球这种结构化网格数据，保留了气象场在三维空间中的物理关联性

    在气象预测中的实际效果：
    每个4°x4°区块会独立产生192维特征表示
    垂直方向2层的气压数据被融合处理
    最终输出形状为：(batch, 8, 360, 181, 192) 其中8是垂直维度降采样后的层数(12/2+2/2(自动padding)+1地层面)
    """
    self.conv_surface = Conv2d(input_dims=7, output_dims=dim, kernel_size=patch_size[1:], stride=patch_size[1:])

    # Load constant masks from the disc
    self.land_mask, self.soil_type, self.topography = LoadConstantMask()
    
  def forward(self, input, input_surface):
    # Zero-pad the input 对输入的三维气象数据进行零填充(padding)操作。
    # 确保卷积操作后特征图的空间尺寸正确
    # 处理边界区域的气象数据，避免因卷积导致的有效区域缩小
    # pad的具体使用示例：
    # pad = [1, 0, 1, 2, 0, 0]       # 填充参数：[width左, width右, height上, height下, depth前, depth后]
    # mode = "constant"               # 填充模式：用常数填充（默认0）
    # my_pad = nn.Pad3D(padding=pad, mode=mode)
    #             [0. 0. 0. 0.]
    # [1. 2. 3.]  [0. 1. 2. 3.]
    # [4. 5. 6.]->[0. 4. 5. 6.]
    #             [0. 0. 0. 0.]
    #             [0. 0. 0. 0.]
    input = Pad3D(input)
    input_surface = Pad2D(input_surface)

    # Apply a linear projection for patch_size[0]*patch_size[1]*patch_size[2] patches, patch_size = (2, 4, 4) as in the original paper
    # 通过3D卷积操作将原始气象数据映射到特征空间，卷积核的权重矩阵实际上就是线性变换矩阵
    # 卷积结果形状计算(二维情况): Output = (Input-Kernel+2*Padding)/Stride+1 （括号内的计算结果向下取整），流程为：输入尺寸减去卷积核尺寸、除以步长、向下取整并补偿首次覆盖的区域(最后+1)
    # 垂直维度：13层 → 6+1个patch（每patch含2层，最后一层不足时自动padding），水平维度：721×1440 → 181×360（每个4°×4°网格为一个patch）
    input = self.conv(input)

    # Add three constant fields to the surface fields
    # 将原始地表气象数据与三个静态地理特征(陆地掩膜、土壤类型、地形高度)合并，为模型提供更全面的地表信息
    # INPUT SHAPE: input_surface = (1440, 721, 4)
    # OUTPUT SHAPE: input_surface = (1440, 721, 4+3)
    # Concatenate默认是沿第一个维度拼接，如
    # a=np.array([[1,2,3],[4,5,6]]) shape:(2,3)
    # b=np.array([[11,21,31],[7,8,9]]) shape:(2,3)
    # c = np.concatenate((a,b))
    # c shape: (4,3)
    # 此处应该沿最后一个维度拼接，即Concatenate((input_surface, self.land_mask, self.soil_type, self.topography),axis=1)
    # Question: 拼接后的张量到底如何满足self.conv_surface中input_dims=7的限制条件的？通过何种手段进行的维度匹配与判定？
    # Answer: 在Pytorch中，输入数据的维度组织一般为[B,C,H,W],因此卷积核的维度设置为[n,c,h2,w2]
    # 其中: B: batch size, C: channel, 卷积核与数据的通道数相等，即问题中的input_dims=7
    # 注意: 在多输入通道的情况下，当输入数组的通道数与卷积核的通道数相同时，在每个通道上，二维的输入数组会与二维核数组做互相关运算，再按通道相加得到输出(因此最终的输出一定为单通道)
    # 如果需要输出数组为多通道，则需要为每个输出通道都分别创建形状为[c,h2,w2]的核数组，卷积核的形状即为[n,c,h2,w2]
    # 因此，n代表了核数组数量，对应着输出通道数，输出张量的维度信息为[b,n,h',w'] Question:h'和w'是否等于h2和w2？ Answer:并非相等关系，h'和w'具体数值依据卷积结果而定，仅在特殊情况(如输入数组每个通道为3*3，而卷积核为2*2时)相等
    input_surface =  Concatenate(input_surface, self.land_mask, self.soil_type, self.topography)

    # Apply a linear projection for patch_size[1]*patch_size[2] patches
    input_surface = self.conv_surface(input_surface)

    # Concatenate the input in the pressure level, i.e., in Z dimension
    x = Concatenate(input, input_surface)

    # 输入x的原始维度结构（卷积拼接后的结果）
    # 假设原始形状：[B, C=192, Z=8, H=360, W=181]
    # 其中：
    # B: batch size
    # C: 通道数（特征维度）
    # Z: 垂直层数（气压层）
    # H: 纬度网格数
    # W: 经度网格数
    # Reshape x for calculation of linear projections
    x = TransposeDimensions(x, (0, 2, 3, 4, 1))
    # 转置后的维度结构：
    # [B, Z=8, H=360, W=181, C=192]
    # 其中192是每个patch的特征维度，8*360*181是patch的数量
    # 转换为二维张量(空间信息序列化)：将三维空间结构（Z×H×W）展开为二维序列（空间位置×特征）
    # 可理解为将每个空间位置（8×360×181=1,046,880个位置）转换为序列元素，192维特征作为每个空间位置的嵌入表示
    x = reshape(x, target_shape=(x.shape[0], 8*360*181, x.shape[-1]))  # 连接卷积特征与Transformer处理
    return x
