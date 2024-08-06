# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: antsfm
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 实现示意图
#
# ![](images/640.webp)

# %% [markdown]
# # 导入库和工具函数

# %%
import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
from torchtext.data import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import spacy
import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


# Set to False to skip notebook execution (e.g. for debugging)
warnings.filterwarnings("ignore")
RUN_EXAMPLES = True

# %%
# Some convenience helper functions used throughout the notebook
alt.renderers.enable("mimetype")  # 解决 Altair 渲染问题


def is_interactive_notebook():
    return __name__ == "__main__"


def show_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        return fn(*args)

def execute_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        fn(*args)

# 占位优化器。推理的时候用来传参占位置
class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


# 占位规划器（规划学习率）。推理的时候用来传参占位置
class DummyScheduler:
    def step(self):
        None


# %% [markdown]
# # 代码结构
#
# 这里面，通常二维张量的形状可能是 (seq_len, d_model)、(batch_size, seq_len)，三维张量的形状可能是 (batch_size, seq_len, d_model)
#
# 训练的时候，seq_len 可能就是固定的。推理的时候，seq_len 可能是 1、2、3、……
#
# 训练的时候，batch_size 可能还是 1.
#
# ## Embedding
#
# 论文说，In our model, we share the same weight matrix between the two embedding layers and
# the pre-softmax linear transformation, similar to [(cite)](https://arxiv.org/abs/1608.05859). 然而，输入层和输出层的 vocab_size 很可能不同，所以这种情况下就不太可能有相同的权值矩阵。
#
# 说到权值矩阵，这玩意显然可以学习。
#
# 论文说，In the embedding layers, we multiply those weights by $\sqrt{d_{\text{model}}}$. 代码里也确实是这么写的。

# %%
class Embeddings(nn.Module):
    """
    直接调 torch 的 embedding
    d_model 是 dim of model 的意思，也就是论文中的 512
    lut 是 lookup table 的意思
    """

    def __init__(self, d_model: int, vocab_size: int):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor):
        """
        输出是一个高维向量张量，形状与输入张量相同，但最后加上一个嵌入维度
        如果输入张量的形状是 (batch_size, sequence_length)
        输出张量的形状是 (batch_size, sequence_length, d_model)
        """
        return self.lut(x) * math.sqrt(self.d_model)


# %% [markdown]
# ## 位置编码
#
# In this work, we use sine and cosine functions of different frequencies:
#
# $$PE_{(pos,2i)} = \sin(pos / 10000^{2i/d_{\text{model}}})$$
#
# $$PE_{(pos,2i+1)} = \cos(pos / 10000^{2i/d_{\text{model}}})$$
#
# $pos / 10000^{2i/d_{\text{model}}} = pos \times 10000^{-2i/d_{\text{model}}}$。我们可以计算 $10000^{-2i/d_{\text{model}}}$ by $\exp(\log(10000^{-2i/d_{\text{model}}})) = \exp(-2i\cdot\log(10000)/d_{\text{model}})$。其中 $2i$ 的取值就是 $0,2,\ldots,d_{\text{model}}-2$
#
# where $pos$ is the position and $i$ is the dimension.  That is, each
# dimension of the positional encoding corresponds to a sinusoid.  The
# wavelengths form a geometric progression from $2\pi$ to $10000 \cdot
# 2\pi$.  We chose this function because we hypothesized it would
# allow the model to easily learn to attend by relative positions,
# since for any fixed offset $k$, $PE_{pos+k}$ can be represented as a
# linear function of $PE_{pos}$. 尽管训练集的序列长度有限，我们也容许长一点的序列，但是还是不能太长，不然位置编码不能预生成得现算了。
#
# In addition, we apply dropout to the sums of the embeddings and the
# positional encodings in both the encoder and decoder stacks.  For
# the base model, we use a rate of $P_{drop}=0.1$. 所以，最后要加 drop

# %%
class PositionalEncoding(nn.Module):
    """
    Implement the PE function.
    forward 输入张量形状是 (batch_size, seq_len, d_model)，即 (batch, seq, feature)
    """

    def __init__(self, d_model: int, dropout_rate: float, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)  # 创建 dropout 层
        # 第一维序列下标，第二维 d_model
        position = torch.arange(0, max_len).unsqueeze(1)  # 变成 (max_len, 1) 形状
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # 注意广播机制和 hadamard 积
        pe[:, 1::2] = torch.cos(position * div_term)  # 注意广播机制和 hadamard 积
        pe = pe.unsqueeze(0)  # pe 变成 (1, max_len, d_model) 形状
        self.register_buffer("pe", pe)  # 这是个缓冲区，不是个参数

    def forward(self, x: torch.Tensor):
        # 和 pe 通过广播机制相加
        # 第二个维度的大小确定好，第三个维度不变
        # 防止 pe 计算梯度
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


# %% [markdown]
# 测试位置编码

# %%
def example_positional():
    pe = PositionalEncoding(20, 0)
    y = pe.forward(torch.zeros(1, 100, 20))

    data = pd.concat(
        [
            pd.DataFrame(
                {
                    "embedding": y[0, :, dim],
                    "dimension": dim,
                    "position": list(range(100)),
                }
            )
            for dim in [4, 5, 6, 7]
        ]
    )

    return (
        alt.Chart(data)
        .mark_line()
        .properties(width=800)
        .encode(x="position", y="embedding", color="dimension:N")
        .interactive()
    )


show_example(example_positional)


# %% [markdown]
# ## 注意力机制
#
# 便捷地克隆模块：

# %%
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# %% [markdown]
# 单头注意力（缩放点积注意力，也就是V的参数是缩放点积的softmax）
#
# $$ \mathrm{Attention}(Q, K, V) = \mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V $$
#
# ![](images/ModalNet-19.png)

# %%
def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask=None,
    dropout=None,
):
    """
    传入：Q、K、V
    mask：要保留的为 1，不保留的为 0
    dropout：一个 dropout 层
    """
    d_k = query.size(-1)  # 张量最后一个维度大小，即 d_model
    # 转置：交换后两个维度
    # 如果这两个矩阵都是 (a, b, c) 维，那么其实相当于做了个 batch=a 的矩阵乘法
    # 如果维度更高也行，反正最后两维是矩阵的行、列
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 在训练decoder的时候，我们不想让decoder位置靠前的看到位置靠后的
    # 这个masked多头注意力，就是在计算出来注意力评分以后softmax之前，把不想要的v的评分置为负无穷
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


# %% [markdown]
# 多头注意力
#
# $$
# \mathrm{MultiHead}(Q, K, V) =
#     \mathrm{Concat}(\mathrm{head_1}, ..., \mathrm{head_h})W^O \\
#     \text{where}~\mathrm{head_i} = \mathrm{Attention}(QW^Q_i, KW^K_i, VW^V_i)
# $$
#
# Where the projections are parameter matrices $W^Q_i \in
# \mathbb{R}^{d_{\text{model}} \times d_k}$, $W^K_i \in
# \mathbb{R}^{d_{\text{model}} \times d_k}$, $W^V_i \in
# \mathbb{R}^{d_{\text{model}} \times d_v}$ and $W^O \in
# \mathbb{R}^{hd_v \times d_{\text{model}}}$.
#
# mask 对于 tgt 来说，是 (batch_size, seq_len, seq_len)。对于 src 来说，是 (batch_size, 1, src_seq_len)。src 的 mask 主要是为了遮住 padding。我认为，src 的第 1 维大小是 1 是为了广播到合适的 src_seq_len。因为 decoder 中的注意力有一层是 query 来自 tgt，key/value 来自 src。

# %%
class MultiHeadedAttention(nn.Module):
    """
    h：头数。必须能整除 d_model
    """

    def __init__(self, h: int, d_model: int, dropout_rate=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0  #
        self.d_k = d_model // h  # key 的维度。也用作 value 的维度
        self.h = h
        # 四个线性变换，前三个用于 Q、K、V 分头投影
        # 第四个用于 concat 后的线性变换（投影矩阵 W^O）
        # 分成 h x d_k 个矩阵就是分成 1 个 d_model 个矩阵
        # 所以这四个线性层变换还可以一样
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None  # 把多头注意力保存下来后面可以可视化
        self.dropout = nn.Dropout(dropout_rate)  # 这玩意也没参数，公用一个也行

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None
    ):
        nbatches = query.size(0)

        # 1. 线性映射。把 d_model 的 query/key/value -> h x d_k
        # 即 (batch_size, seq_len, d_model) -> (batch_size, seq_len, h, d_k)
        # 再转置一把，变成 (batch_size, h, seq_len, d_k)
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]
        # 2. mask 扩充。(batch_size, seq_len, seq_len) -> (batch_size, 1, seq_len, seq_len)
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        # 3. 多头注意力计算
        x, self.attn = attention(query, key, value, mask, self.dropout)
        # 4. concat
        # 复原张量。把内存变连续，然后才能 view
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        del query
        del key
        del value
        # 5. 线性计算
        return self.linears[-1](x)


# %% [markdown]
# ### 生成 mask
#
# mask 一般都是为 1 的留下

# %%
def subsequent_mask(size: int):
    "Mask out subsequent positions."
    "形如(1, size, size)"
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    # 生成符合形状的下三角
    return subsequent_mask == 0


# %%
def example_mask():
    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Subsequent Mask": subsequent_mask(20)[0][x, y].flatten(),
                    "Window": y,
                    "Masking": x,
                }
            )
            for y in range(20)
            for x in range(20)
        ]
    )

    return (
        alt.Chart(LS_data)
        .mark_rect()
        .properties(height=250, width=250)
        .encode(
            alt.X("Window:O"),
            alt.Y("Masking:O"),
            alt.Color("Subsequent Mask:Q", scale=alt.Scale(scheme="viridis")),
        )
        .interactive()
    )


show_example(example_mask)


# %% [markdown]
# ## 前馈网络
#
# 就是俩线性层

# %%
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model: int, d_ff: int, dropout_rate=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor):
        return self.w_2(self.dropout(self.w_1(x).relu()))  # 这么写 relu 也行


# %% [markdown]
# ## 层归一化、残差连接，及子层连接
#
# 层归一化里面也有可学习参数
#
# 和论文里不太一样，这里是先层归一化，再子层，再 dropout，再残差

# %%
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    """
    对最后一维（feature 维）做层归一化
    """

    def __init__(self, features_size: int, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features_size))  # 可学习参数
        self.b_2 = nn.Parameter(torch.zeros(features_size))  # 可学习参数
        self.eps = eps

    def forward(self, x: torch.Tensor):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# %%
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    size 就是 d_model
    也就是多头注意力/掩码多头注意力/前馈网络的 wrapper
    """

    def __init__(self, size: int, dropout_rate: float):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)

    # sublayer 可能是 nn.Module，也可能是 lambda 函数
    # 总之能通过 () 调用就行
    def forward(self, x: torch.Tensor, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


# %% [markdown]
# ## 一层 encoder
#
#

# %%
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    """
    传入多头注意力模块和前馈模块
    """

    def __init__(self, size: int, self_attn, feed_forward, dropout_rate: float):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # encoder 有两层，一层多头注意力，一头前馈网络
        self.sublayer_connections = clones(SublayerConnection(size, dropout_rate), 2)
        self.size = size

    def forward(self, x: torch.Tensor, mask):
        # 自注意力。要传入一个 lambda 供调用
        x = self.sublayer_connections[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer_connections[1](x, self.feed_forward)


# %% [markdown]
# ## 一层 decoder
#
# 简直跟 encoder 一模一样！就是多了一层

# %%
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    """
    传入多头注意力模块和前馈模块
    """

    def __init__(
        self, size: int, self_attn, src_attn, feed_forward, dropout_rate: float
    ):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout_rate), 3)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        "tgt_mask 是给 decoder 那里用的"
        "src_mask 是给 encoder 到 decoder 那里的 mask 用的"
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


# %% [markdown]
# ## 6 层 encoder 堆叠成一个 encoder

# %%
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    # layer 应该是 EncoderLayer
    def __init__(self, layer, N: int):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: torch.Tensor, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# %% [markdown]
# ## 6 层 decoder 堆叠成一个 decoder

# %%
class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N: int):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# %% [markdown]
# ## 最后的 generator：又生成词的概率

# %%
class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    "最后一维"

    def __init__(self, d_model: int, vocab_size: int):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor):
        return log_softmax(self.proj(x), dim=-1)


# %% [markdown]
# ## 终极无敌最终模型

# %%
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    encoder：Encoder
    decoder：Decoder
    src_embed：在 src 起到 embedding 和位置编码的模块
    dst_embed：在 dst 起到 embedding 和位置编码的模块
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        src_embed: nn.Module,
        tgt_embed: nn.Module,
        generator: nn.Module,
    ):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src: torch.Tensor, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory: torch.Tensor, src_mask, tgt: torch.Tensor, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)


# %% [markdown]
# # 构建模型
#
# ## The Model

# %%
def make_model(
    src_vocab: int, tgt_vocab: int, N=6, d_model=512, d_ff=2048, h=8, dropout_rate=0.1
):
    "Helper: Construct a model from hyperparameters."
    "源/目的词汇表大小、堆叠层数、模型大小、前馈层大小、头数、dropout 率"
    cpy = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, dropout_rate)  # 后面都要深拷贝使用
    ff = PositionwiseFeedForward(d_model, d_ff, dropout_rate)  # 后面都要深拷贝使用
    position = PositionalEncoding(d_model, dropout_rate)  # 后面都要深拷贝使用
    model = EncoderDecoder(
        encoder=Encoder(EncoderLayer(d_model, cpy(attn), cpy(ff), dropout_rate), N),
        decoder=Decoder(
            DecoderLayer(d_model, cpy(attn), cpy(attn), cpy(ff), dropout_rate), N
        ),
        src_embed=nn.Sequential(Embeddings(d_model, src_vocab), cpy(position)),
        tgt_embed=nn.Sequential(Embeddings(d_model, tgt_vocab), cpy(position)),
        generator=Generator(d_model, tgt_vocab),
    )
    # 重要初始化
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# %% [markdown]
# ## 模拟推理

# %%
def inference_test():
    test_model = make_model(11, 11, 2)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print("Example Untrained Model Prediction:", ys)


def run_tests():
    for _ in range(10):
        inference_test()


show_example(run_tests)


# %% [markdown]
# # 模型训练
#
# ## batch
#
# 一个 batch 中的数据、mask 由 Batch 对象代表。

# %%
class Batch:
    """
    在训练/推理时，保存传入的 tgt 和 src 并生成 mask、token 数等等
    src, tgt 应该都是 (batch_size, seq_size)。值是词语的编号
    src_mask shape (batch_size, 1, src_seq_size)
    tgt_mask shape (batch_size, tgt_seq_size, tgt_seq_size)
    """

    @staticmethod
    def make_tgt_mask(tgt: torch.Tensor, pad: int):
        "Create a mask to hide padding and future words."
        "pad：就是 <blank>，即填充。当然应该 mask 掉"
        tgt_mask = (tgt != pad).unsqueeze(
            -2
        )  # 这时候第 1 维大小还是 1（维序号从 0 开始）。马上广播
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        # 首先去除掉目的为 padding 的，然后再套上三角 mask。中间经过了广播
        return tgt_mask

    """Object for holding a batch of data with mask during training."""

    def __init__(self, src: torch.Tensor, tgt=None, pad=2):  # 2 = <blank>
        """
        src 必传入。因为训练和推理显然都有 src
        tgt 训练时传入，推理时不传入
        pad 是”填充“ <blank> 的序号，应该被 mask 掉
        """
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            # 生成 tgt 序列和 tgt_y：前一个词生成后一个词
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_tgt_mask(self.tgt, pad)
            self.ntokens = (
                self.tgt_y != pad
            ).data.sum()  # 记录这一批里面非填充 token 有多少个，供后面计算速度


# %% [markdown]
# ## 一 epoch 训练

# %%
class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


def run_epoch(
    data_iter,  # 生成器，生成在循环中调用它可以生成各个 batch 的数据
    model: EncoderDecoder,
    loss_compute,  # 接收预测结果、正确 target、tokens 数，返回用于输出的 loss 值和用于反向传播的 loss_node
    optimizer,  # 优化器。训练时使用，预测时传 Dummy
    scheduler,  # 调度器。用来调整学习率
    mode="train",
    accum_iter=1,  # 多少个 batch 增加一次梯度
    train_state=TrainState(),
):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        # 这里的 out 还只是 (batch_size, seq_size, d_model)，没有进 generator
        # 是在 loss_compute 里面进的 generator
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()  # 用于调整学习率
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state


# %% [markdown]
# ## 学习率变化
#
# We used the Adam optimizer [(cite)](https://arxiv.org/abs/1412.6980)
# with $\beta_1=0.9$, $\beta_2=0.98$ and $\epsilon=10^{-9}$.  We
# varied the learning rate over the course of training, according to
# the formula:
#
# $$
# lrate = d_{\text{model}}^{-0.5} \cdot
#   \min({step\_num}^{-0.5},
#     {step\_num} \cdot {warmup\_steps}^{-1.5})
# $$
#
# 使用 LambdaLR（lambda learning rate）更新每个 step 后 optimizer 的学习率

# %%
def rate(step: int, model_size: int, factor: float, warmup: int):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


# %%
def example_learning_schedule():
    opts = [
        [512, 1, 4000],  # example 1
        [512, 1, 8000],  # example 2
        [256, 1, 4000],  # example 3
    ]

    dummy_model = torch.nn.Linear(1, 1)
    learning_rates = []

    # we have 3 examples in opts list.
    for idx, example in enumerate(opts):
        # run 20000 epoch for each example
        optimizer = torch.optim.Adam(
            dummy_model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9
        )
        lr_scheduler = LambdaLR(
            optimizer=optimizer, lr_lambda=lambda step: rate(step, *example)
        )
        tmp = []
        # take 20K dummy training steps, save the learning rate at each step
        for step in range(20000):
            tmp.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            lr_scheduler.step()
        learning_rates.append(tmp)

    learning_rates = torch.tensor(learning_rates)

    # Enable altair to handle more than 5000 rows
    alt.data_transformers.disable_max_rows()

    opts_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Learning Rate": learning_rates[warmup_idx, :],
                    "model_size:warmup": ["512:4000", "512:8000", "256:4000"][
                        warmup_idx
                    ],
                    "step": range(20000),
                }
            )
            for warmup_idx in [0, 1, 2]
        ]
    )

    return (
        alt.Chart(opts_data)
        .mark_line()
        .properties(width=600)
        .encode(x="step", y="Learning Rate", color="model_size\:warmup:N")
        .interactive()
    )


example_learning_schedule()


# %% [markdown]
# ## 标签平滑正则化（损失函数）
#
# 目的：传入很多预测概率和真正的 target，计算损失函数。
#
# 通常来说，真正概率要么 1.0 要么 0.0。平滑一下就是真正标签的概率 1-smoothing，其他标签的概率 smoothing / (size - 2)
#
# 为什么是减 2 呢？减去一个真正标签还要减去一个“填充”（padding/blank）。填充是不分配概率的
#

# %%
class LabelSmoothingLoss(nn.Module):
    "Implement label smoothing."
    """
    size：类别大小
    padding_idx：“填充”标签的编号（即后面的 <blank>）。这个不分配概率
    smoothing：平滑系数
    """

    def __init__(self, size: int, padding_idx: int, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")  # 损失函数
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing  # 置信度
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None  # target 平滑后的概率记录下来

    # 希望传入：x (batch_size, size)，代表一个 batch 对每个类别预测的概率
    # target：(batch_size,) 代表这个 batch 的真实类别
    # 返回损失函数
    # 输入好像需要 x 经过了 log，并且没有 0.0
    def forward(self, x: torch.Tensor, target: torch.Tensor):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))  # 先设置 size - 2 个默认值
        true_dist.scatter_(
            1, target.data.unsqueeze(1), self.confidence
        )  # 设置真实类别。unsqueeze 让维度匹配
        true_dist[:, self.padding_idx] = 0  # 置零”填充“标签
        # 对于 target 为 padding_idx 的行，把那一行都设置成概率为 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist  # 保存下来 target 平滑以后的概率
        return self.criterion(x, true_dist.clone().detach())  # 计算损失函数


# %% [markdown]
# 展现标签平滑损失函数的标签平滑效果的例子。在下面这个例子中，batch_size=6。padding_idx=0。有5个类别。
#
# 展示的是 target 平滑以后的概率（这个将要和预测概率进行损失函数计算）。可以看到，正确的类别概率是 0.6，错误的平分 0.4 的概率（填充类不分配）。如果 target 就是填充类，target 概率全设置成 0。

# %%
def example_label_smoothing():
    crit = LabelSmoothingLoss(5, 0, 0.4)
    predict = torch.FloatTensor(
        [
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
        ]
    )
    crit(x=predict.log(), target=torch.LongTensor([2, 1, 0, 3, 3, 4]))
    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "target distribution": crit.true_dist[x, y].flatten(),
                    "columns": y,
                    "rows": x,
                }
            )
            for y in range(5)
            for x in range(6)
        ]
    )

    return (
        alt.Chart(LS_data)
        .mark_rect(color="Blue", opacity=1)
        .properties(height=200, width=200)
        .encode(
            alt.X("columns:O", title=None),
            alt.Y("rows:O", title=None),
            alt.Color("target distribution:Q", scale=alt.Scale(scheme="viridis")),
        )
        .interactive()
    )


show_example(example_label_smoothing)


# %% [markdown]
# 展现标签平滑损失函数的损失函数的例子。

# %%
def loss(x, crit):
    d = x + 3 * 1
    # predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d]])
    predict = torch.FloatTensor(
        [[1e-9, x / d - 1e-9, 1 / d, 1 / d, 1 / d]]
    )  # 防止出现 NaN
    return crit(predict.log(), torch.LongTensor([1])).data


def penalization_visualization():
    crit = LabelSmoothingLoss(5, 0, 0.1)
    loss_data = pd.DataFrame(
        {
            "Loss": [loss(x, crit) for x in range(1, 100)],
            "Steps": list(range(99)),
        }
    ).astype("float")

    return (
        alt.Chart(loss_data)
        .mark_line()
        .properties(width=350)
        .encode(
            x="Steps",
            y="Loss",
        )
        .interactive()
    )


show_example(penalization_visualization)


# %% [markdown]
# # 简单的例子：原样输出
#
# 训练一个原样输出器

# %%
def data_gen(V: int, batch_size: int, nbatches: int):
    "Generate random data for a src-tgt copy task."
    "V 是 词汇表大小。序列长为 10"
    for i in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))  # 防止生成到 0
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0)


# %% [markdown]
# 简单的损失计算
#

# %%
class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator: nn.Module, criterion: nn.Module):
        """
        generator 是根据 (batch_size, seq_size, d_model) 生成 (batch_size, seq_size, vocab_size)
        criterion 是根据预测的概率和 tgt 计算损失函数的模块
        """
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x: torch.Tensor, y: torch.Tensor, norm):
        """
        norm 大概是一个系数，用来展示的时候大一点
        """
        x = self.generator(x)
        sloss = (
            # 不管你几维，都变成二维
            self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1))
            / norm
        )
        return sloss.data * norm, sloss


# %% [markdown]
# 贪心 decoder（模拟推理）

# %%
def greedy_decode(
    model: EncoderDecoder,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
):
    memory = model.encode(src, src_mask) # 先 encode
    # 构造初始生成序列 y sequences 或者叫 output sequences
    # 起初，(batch_size, seq_len) = (1, 1)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    # 一个一个生成词语
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys


# %% [markdown]
# 训练原样输出模型并推理

# %%
# Train the simple copy task.
def example_simple_model():
    V = 11
    criterion = LabelSmoothingLoss(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400
        ),
    )

    batch_size = 80
    for epoch in range(20):
        model.train()
        run_epoch(
            data_gen(V, batch_size, 20),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train",
        )
        model.eval()
        run_epoch(
            data_gen(V, batch_size, 5),
            model,
            SimpleLossCompute(model.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )[0]

    model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len)
    print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))


# execute_example(example_simple_model)

# %% [markdown]
# # 真正例子
#
# 分词器、数据集下载
#

# %%
# Load spacy tokenizer models, download them if they haven't been
# downloaded already


def load_tokenizers():

    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en

def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]


def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])

def my_multi30k_filter_fn(split, language_pair, i, x):
    return f"/{datasets.multi30k._PREFIX[split]}.{language_pair[i]}" in x[0]
datasets.multi30k._filter_fn = my_multi30k_filter_fn
# https://github.com/pytorch/text/issues/2221 to fix invalid unicode error
# 原来的包会错误使用 ._test.de 这个 APPLE 自动保存的烂东西

def build_vocabulary(spacy_de, spacy_en):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    print("Building German Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_de, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    print("Building English Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_en, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt


def load_vocab(spacy_de, spacy_en):
    if not exists("vocab.pt"):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
        torch.save((vocab_src, vocab_tgt), "vocab.pt")
    else:
        vocab_src, vocab_tgt = torch.load("vocab.pt")
    print("Finished.\nVocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt


if is_interactive_notebook():
    # global variables used later in the script
    # 两个分词器
    spacy_de, spacy_en = show_example(load_tokenizers)
    # 两个词汇表。词汇表可以把一个 list[str] 转换成 list[int]
    vocab_src, vocab_tgt = show_example(load_vocab, args=[spacy_de, spacy_en]) 


# %% [markdown]
# 词语定序（把句子通过词汇表转换成为整数序列）

# %%
def collate_batch(
    batch, # batch 是 [(de_sentence,en_sentence)]。是 DataLoader 传进来的类型
    src_pipeline, # 把一个句子分词分成 list of str
    tgt_pipeline, # 把一个句子分词分成 list of str
    src_vocab, # src 的词汇表
    tgt_vocab, # tgt 的词汇表
    device,
    max_padding=128, # 最大填充到多长
    pad_id=2, # <blank> 这个填充元素的 id
):
    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id
    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    src_vocab(src_pipeline(_src)), # 先分词再词汇表
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    tgt_vocab(tgt_pipeline(_tgt)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        src_list.append(
            # warning - overwrites values for negative values of padding - len
            pad(
                processed_src,
                (
                    0, # 前面不填充，尾部填充
                    max_padding - len(processed_src),
                ),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )

    src = torch.stack(src_list) # 堆成一个 (batch_size, seq_len) 的张量
    tgt = torch.stack(tgt_list)
    return (src, tgt)


# %% [markdown]
# 创建 dataloader

# %%
def create_dataloaders(
    device,
    vocab_src, # src 词汇表
    vocab_tgt, # tgt 词汇表
    spacy_de, # 德语 spacy 对象，用于创建分词器
    spacy_en, # 英语 spacy 对象，用于创建分词器
    batch_size=12000, # 靠，这么大
    max_padding=128, # 最大填充长度
    is_distributed=True,
):
    # def create_dataloaders(batch_size=12000):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    def collate_fn(batch): # 传给 dataloader
        return collate_batch(
            batch,
            tokenize_de,
            tokenize_en,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    train_iter, valid_iter, test_iter = datasets.Multi30k(
        language_pair=("de", "en")
    )

    train_iter_map = to_map_style_dataset(
        train_iter
    )  # DistributedSampler needs a dataset len()
    train_sampler = (
        DistributedSampler(train_iter_map) if is_distributed else None
    )
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = (
        DistributedSampler(valid_iter_map) if is_distributed else None
    )

    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader


# %% [markdown]
# train worker
#
# 并行多卡计算的代码就是这种，没啥特别好说的

# %%
def train_worker(
    gpu,
    ngpus_per_node,
    vocab_src, # src 词汇表
    vocab_tgt, # tgt 词汇表
    spacy_de, # 德语 spacy 对象，用于创建分词器
    spacy_en, # 英语 spacy 对象，用于创建分词器
    config, # 自己传入的字典
    is_distributed=False,
):
    print(f"Train worker process using GPU: {gpu} for training", flush=True)
    torch.cuda.set_device(gpu)

    pad_idx = vocab_tgt["<blank>"]
    d_model = 512
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.cuda(gpu)
    module = model
    is_main_process = True
    if is_distributed:
        dist.init_process_group(
            "nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node
        )
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0

    criterion = LabelSmoothingLoss(
        size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1
    )
    criterion.cuda(gpu)

    train_dataloader, valid_dataloader = create_dataloaders(
        gpu,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=config["batch_size"] // ngpus_per_node,
        max_padding=config["max_padding"],
        is_distributed=is_distributed,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, d_model, factor=1, warmup=config["warmup"]
        ),
    )
    train_state = TrainState()

    for epoch in range(config["num_epochs"]):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)
        _, train_state = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )

        GPUtil.showUtilization()
        if is_main_process:
            file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
            torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()

        print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)
        model.eval()
        sloss = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        print(sloss)
        torch.cuda.empty_cache()

    if is_main_process:
        file_path = "%sfinal.pt" % config["file_prefix"]
        torch.save(module.state_dict(), file_path)


# %% [markdown]
# 训练、保存

# %%
def train_distributed_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    from my_transformer import train_worker
    ngpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    print(f"Number of GPUs detected: {ngpus}")
    print("Spawning training processes ...")
    mp.spawn(
        train_worker,
        nprocs=ngpus,
        args=(ngpus, vocab_src, vocab_tgt, spacy_de, spacy_en, config, True),
    )


def train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    if config["distributed"]:
        train_distributed_model(
            vocab_src, vocab_tgt, spacy_de, spacy_en, config
        )
    else:
        train_worker(
            0, 1, vocab_src, vocab_tgt, spacy_de, spacy_en, config, False
        )


def load_trained_model():
    config = {
        "batch_size": 32,
        "distributed": True, # poorpool modify from false to True
        "num_epochs": 8,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 3000,
        "file_prefix": "multi30k_model_",
    }
    model_path = "multi30k_model_final.pt"
    if not exists(model_path):
        train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)

    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(torch.load("multi30k_model_final.pt"))
    return model


if is_interactive_notebook():
    model = load_trained_model()


# %% [markdown]
# # 看看结果

# %%
def check_outputs(
    valid_dataloader,
    model,
    vocab_src,
    vocab_tgt,
    n_examples=15,
    pad_idx=2,
    eos_string="</s>",
):
    results = [()] * n_examples
    for idx in range(n_examples):
        print("\nExample %d ========\n" % idx)
        b = next(iter(valid_dataloader))
        rb = Batch(b[0], b[1], pad_idx)
        greedy_decode(model, rb.src, rb.src_mask, 64, 0)[0]

        src_tokens = [
            vocab_src.get_itos()[x] for x in rb.src[0] if x != pad_idx
        ]
        tgt_tokens = [
            vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x != pad_idx
        ]

        print(
            "Source Text (Input)        : "
            + " ".join(src_tokens).replace("\n", "")
        )
        print(
            "Target Text (Ground Truth) : "
            + " ".join(tgt_tokens).replace("\n", "")
        )
        model_out = greedy_decode(model, rb.src, rb.src_mask, 72, 0)[0]
        model_txt = (
            " ".join(
                [vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx]
            ).split(eos_string, 1)[0]
            + eos_string
        )
        print("Model Output               : " + model_txt.replace("\n", ""))
        results[idx] = (rb, src_tokens, tgt_tokens, model_out, model_txt)
    return results


def run_model_example(n_examples=5):
    global vocab_src, vocab_tgt, spacy_de, spacy_en

    print("Preparing Data ...")
    _, valid_dataloader = create_dataloaders(
        torch.device("cpu"),
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=1,
        is_distributed=False,
    )

    print("Loading Trained Model ...")

    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(
        torch.load("multi30k_model_final.pt", map_location=torch.device("cpu"))
    )

    print("Checking Model Outputs:")
    example_data = check_outputs(
        valid_dataloader, model, vocab_src, vocab_tgt, n_examples=n_examples
    )
    return model, example_data


execute_example(run_model_example)


# %%
def mtx2df(m, max_row, max_col, row_tokens, col_tokens):
    "convert a dense matrix to a data frame with row and column indices"
    return pd.DataFrame(
        [
            (
                r,
                c,
                float(m[r, c]),
                "%.3d %s"
                % (r, row_tokens[r] if len(row_tokens) > r else "<blank>"),
                "%.3d %s"
                % (c, col_tokens[c] if len(col_tokens) > c else "<blank>"),
            )
            for r in range(m.shape[0])
            for c in range(m.shape[1])
            if r < max_row and c < max_col
        ],
        # if float(m[r,c]) != 0 and r < max_row and c < max_col],
        columns=["row", "column", "value", "row_token", "col_token"],
    )


def attn_map(attn, layer, head, row_tokens, col_tokens, max_dim=30):
    df = mtx2df(
        attn[0, head].data,
        max_dim,
        max_dim,
        row_tokens,
        col_tokens,
    )
    return (
        alt.Chart(data=df)
        .mark_rect()
        .encode(
            x=alt.X("col_token", axis=alt.Axis(title="")),
            y=alt.Y("row_token", axis=alt.Axis(title="")),
            color="value",
            tooltip=["row", "column", "value", "row_token", "col_token"],
        )
        .properties(height=400, width=400)
        .interactive()
    )


# %%
def get_encoder(model, layer):
    return model.encoder.layers[layer].self_attn.attn


def get_decoder_self(model, layer):
    return model.decoder.layers[layer].self_attn.attn


def get_decoder_src(model, layer):
    return model.decoder.layers[layer].src_attn.attn


def visualize_layer(model, layer, getter_fn, ntokens, row_tokens, col_tokens):
    # ntokens = last_example[0].ntokens
    attn = getter_fn(model, layer)
    n_heads = attn.shape[1]
    charts = [
        attn_map(
            attn,
            0,
            h,
            row_tokens=row_tokens,
            col_tokens=col_tokens,
            max_dim=ntokens,
        )
        for h in range(n_heads)
    ]
    assert n_heads == 8
    return alt.vconcat(
        charts[0]
        # | charts[1]
        | charts[2]
        # | charts[3]
        | charts[4]
        # | charts[5]
        | charts[6]
        # | charts[7]
        # layer + 1 due to 0-indexing
    ).properties(title="Layer %d" % (layer + 1))


# %%
def viz_encoder_self():
    model, example_data = run_model_example(n_examples=1)
    example = example_data[
        len(example_data) - 1
    ]  # batch object for the final example

    layer_viz = [
        visualize_layer(
            model, layer, get_encoder, len(example[1]), example[1], example[1]
        )
        for layer in range(6)
    ]
    return alt.hconcat(
        layer_viz[0]
        # & layer_viz[1]
        & layer_viz[2]
        # & layer_viz[3]
        & layer_viz[4]
        # & layer_viz[5]
    )


show_example(viz_encoder_self)


# %%
def viz_decoder_self():
    model, example_data = run_model_example(n_examples=1)
    example = example_data[len(example_data) - 1]

    layer_viz = [
        visualize_layer(
            model,
            layer,
            get_decoder_self,
            len(example[1]),
            example[1],
            example[1],
        )
        for layer in range(6)
    ]
    return alt.hconcat(
        layer_viz[0]
        & layer_viz[1]
        & layer_viz[2]
        & layer_viz[3]
        & layer_viz[4]
        & layer_viz[5]
    )


show_example(viz_decoder_self)


# %%
def viz_decoder_src():
    model, example_data = run_model_example(n_examples=1)
    example = example_data[len(example_data) - 1]

    layer_viz = [
        visualize_layer(
            model,
            layer,
            get_decoder_src,
            max(len(example[1]), len(example[2])),
            example[1],
            example[2],
        )
        for layer in range(6)
    ]
    return alt.hconcat(
        layer_viz[0]
        & layer_viz[1]
        & layer_viz[2]
        & layer_viz[3]
        & layer_viz[4]
        & layer_viz[5]
    )


show_example(viz_decoder_src)
