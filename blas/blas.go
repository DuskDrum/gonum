// Copyright ©2013 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate ./conversions.bash

package blas

// Flag constants indicate Givens transformation H matrix state.
type Flag int

const (
	Identity    Flag = -2 // H is the identity matrix; no rotation is needed.
	Rescaling   Flag = -1 // H specifies rescaling.
	OffDiagonal Flag = 0  // Off-diagonal elements of H are non-unit.
	Diagonal    Flag = 1  // Diagonal elements of H are non-unit.
)

// SrotmParams contains Givens transformation parameters returned
// by the Float32 Srotm method.
type SrotmParams struct {
	Flag
	H [4]float32 // Column-major 2 by 2 matrix.
}

// DrotmParams contains Givens transformation parameters returned
// by the Float64 Drotm method.
type DrotmParams struct {
	Flag
	H [4]float64 // Column-major 2 by 2 matrix.
}

// Transpose specifies the transposition operation of a matrix.
type Transpose byte

const (
	NoTrans   Transpose = 'N'
	Trans     Transpose = 'T'
	ConjTrans Transpose = 'C'
)

// Uplo specifies whether a matrix is upper or lower triangular.
type Uplo byte

const (
	Upper Uplo = 'U'
	Lower Uplo = 'L'
	All   Uplo = 'A'
)

// Diag specifies whether a matrix is unit triangular.
type Diag byte

const (
	NonUnit Diag = 'N'
	Unit    Diag = 'U'
)

// Side specifies from which side a multiplication operation is performed.
type Side byte

const (
	Left  Side = 'L'
	Right Side = 'R'
)

// Float32 implements the single precision real BLAS routines.
type Float32 interface {
	Float32Level1
	Float32Level2
	Float32Level3
}

// Float32Level1 implements the single precision real BLAS Level 1 routines.
// Float32Level1 接口提供了单精度浮点数向量运算的功能，是 BLAS Level 1 操作的单精度版本。在内存敏感、移动设备或需要大量计算的场景中非常有用。
// 计算信号能量（使用 Sasum）
// 滤波器应用（使用 Sdot 实现卷积）
// 3D 图形变换（使用 Srot）
// 向量归一化（使用 Snrm2）
// 计算损失函数的梯度（使用 Saxpy）
// 计算特征向量的相似度（使用 Sdot）
type Float32Level1 interface {
	// Sdsdot - 双精度累加的点积. 用途：单精度向量点积，但使用双精度累加以提高精度
	//数学公式：$\alpha + \sum_{i=0}^{n-1} x_i y_i$（双精度累加）
	Sdsdot(n int, alpha float32, x []float32, incX int, y []float32, incY int) float32
	// Dsdot - 双精度结果的点积. 用途：单精度向量点积，返回双精度结果以提高精度
	//数学公式：$\sum_{i=0}^{n-1} x_i y_i$（返回 float64）
	Dsdot(n int, x []float32, incX int, y []float32, incY int) float64
	// Sdot - 标准点积 用途：计算两个向量的点积
	//数学公式：$\sum_{i=0}^{n-1} x_i y_i$
	Sdot(n int, x []float32, incX int, y []float32, incY int) float32
	// Snrm2 - 欧几里得范数 用途：计算向量的 L2 范数（欧几里得长度）
	//数学公式：$\sqrt{\sum_{i=0}^{n-1} x_i^2}$
	Snrm2(n int, x []float32, incX int) float32
	// Sasum - 绝对值求和  用途：计算向量各元素绝对值的和（L1 范数）
	//数学公式：$\sum_{i=0}^{n-1} |x_i|$
	Sasum(n int, x []float32, incX int) float32
	// Isamax - 最大绝对值索引  用途：找到向量中绝对值最大元素的索引
	//数学公式：$\text{argmax}_i |x_i|$
	Isamax(n int, x []float32, incX int) int
	// Sswap - 向量交换   用途：交换两个向量的内容
	//数学公式：$x_i \leftrightarrow y_i$
	Sswap(n int, x []float32, incX int, y []float32, incY int)
	// Scopy - 向量复制  用途：将向量 x 复制到 y
	//数学公式：$y_i \leftarrow x_i$
	Scopy(n int, x []float32, incX int, y []float32, incY int)
	// Saxpy - 标量乘加  用途：计算 $y = \alpha x + y$
	//数学公式：$y_i \leftarrow \alpha \times x_i + y_i$
	Saxpy(n int, alpha float32, x []float32, incX int, y []float32, incY int)
	// Srotg - 构造 Givens 旋转 用途：构造 Givens 旋转矩阵的参数
	//输出：c(余弦), s(正弦), r, z 参数
	Srotg(a, b float32) (c, s, r, z float32)
	// Srotmg - 构造改进的 Givens 旋转  用途：构造改进的 Givens 旋转参数（用于带缩放的旋转）
	Srotmg(d1, d2, b1, b2 float32) (p SrotmParams, rd1, rd2, rb1 float32)
	// Srot - 应用 Givens 旋转. 用途：对两个向量应用 Givens 旋转
	//数学公式： $\begin{bmatrix} x_i' \ y_i' \end{bmatrix} = \begin{bmatrix} c & s \ -s & c \end{bmatrix} \begin{bmatrix} x_i \ y_i \end{bmatrix}$
	Srot(n int, x []float32, incX int, y []float32, incY int, c, s float32)
	// Srotm - 应用改进的 Givens 旋转  用途：应用改进的 Givens 旋转（带缩放）
	Srotm(n int, x []float32, incX int, y []float32, incY int, p SrotmParams)
	// Sscal - 向量缩放  用途：用标量缩放向量 $x = \alpha x$
	//数学公式：$x_i \leftarrow \alpha \times x_i$
	Sscal(n int, alpha float32, x []float32, incX int)
}

// Float32Level2 implements the single precision real BLAS Level 2 routines.
// Float32Level2 接口提供了单精度浮点数矩阵-向量运算的功能，是 BLAS Level 2 操作的单精度版本。在内存敏感、移动设备或需要大量矩阵-向量计算的场景中非常有用。
// 机器学习 - 神经网络. 全连接层前向传播. 权重更新（使用外积）
// 计算机图形学 -  // 3D 变换：应用变换矩阵到顶点集合.     // 每个顶点是3D向量，批量应用变换
// 数值优化. // 共轭梯度法中的矩阵-向量乘 .
// 控制系统 卡尔曼滤波预测步骤
type Float32Level2 interface {
	// Sgemv - 通用矩阵-向量乘法
	//用途：计算通用矩阵与向量的乘积
	//数学公式：$y \leftarrow \alpha op(A) x + \beta y$
	//
	//$op(A) = A$ 或 $A^T$
	//
	//支持非转置、转置操作
	Sgemv(tA Transpose, m, n int, alpha float32, a []float32, lda int, x []float32, incX int, beta float32, y []float32, incY int)
	// Sgbmv - 带状矩阵-向量乘法
	// 用途：带状矩阵与向量的乘积（仅存储非零对角线）
	// 参数：
	//kL：下带宽
	//kU：上带宽
	//适用：三对角矩阵等稀疏但结构化的矩阵
	Sgbmv(tA Transpose, m, n, kL, kU int, alpha float32, a []float32, lda int, x []float32, incX int, beta float32, y []float32, incY int)
	// Strmv - 三角矩阵-向量乘法
	// 用途：三角矩阵与向量的乘积
	//数学公式：$x \leftarrow op(A) x$
	//参数：
	//uplo：Upper 或 Lower（上三角或下三角）
	//diag：Unit 或 NonUnit（单位三角矩阵或普通）
	Strmv(ul Uplo, tA Transpose, d Diag, n int, a []float32, lda int, x []float32, incX int)
	// Stbmv - 三角带状矩阵-向量乘法
	// 用途：三角带状矩阵与向量的乘积
	Stbmv(ul Uplo, tA Transpose, d Diag, n, k int, a []float32, lda int, x []float32, incX int)
	// Stpmv - 压缩三角矩阵-向量乘法
	// 用途：压缩存储的三角矩阵与向量的乘积
	Stpmv(ul Uplo, tA Transpose, d Diag, n int, ap []float32, x []float32, incX int)
	// Strsv - 三角方程组求解
	// 用途：解三角方程组 $op(A) x = b$
	//数学公式：$x \leftarrow A^{-1} x$（原地求解）
	//应用：线性方程组求解的关键步骤
	Strsv(ul Uplo, tA Transpose, d Diag, n int, a []float32, lda int, x []float32, incX int)
	// Stbsv - 三角带状方程组求解
	// 用途：解三角带状方程组
	Stbsv(ul Uplo, tA Transpose, d Diag, n, k int, a []float32, lda int, x []float32, incX int)
	// Stpsv - 压缩三角方程组求解
	// 用途：解压缩存储的三角方程组
	Stpsv(ul Uplo, tA Transpose, d Diag, n int, ap []float32, x []float32, incX int)
	// Ssymv - 对称矩阵-向量乘法
	// 用途：对称矩阵与向量的乘积
	//数学公式：$y \leftarrow \alpha A x + \beta y$（A 是对称矩阵）
	//特性：$A = A^T$，只需存储上三角或下三角
	Ssymv(ul Uplo, n int, alpha float32, a []float32, lda int, x []float32, incX int, beta float32, y []float32, incY int)
	// Ssbmv - 对称带状矩阵-向量乘法
	// 用途：对称带状矩阵与向量的乘积
	Ssbmv(ul Uplo, n, k int, alpha float32, a []float32, lda int, x []float32, incX int, beta float32, y []float32, incY int)
	// Sspmv - 压缩对称矩阵-向量乘法
	// 用途：压缩存储的对称矩阵与向量的乘积
	Sspmv(ul Uplo, n int, alpha float32, ap []float32, x []float32, incX int, beta float32, y []float32, incY int)
	// Sger - 外积（秩1更新）
	// 用途：计算秩1更新 $A \leftarrow \alpha x y^T + A$
	//数学公式：$A_{ij} \leftarrow \alpha x_i y_j + A_{ij}$
	//应用：矩阵的低秩更新，神经网络中的外积计算
	Sger(m, n int, alpha float32, x []float32, incX int, y []float32, incY int, a []float32, lda int)
	// Ssyr - 对称秩1更新
	// 用途：对称矩阵的秩1更新 $A \leftarrow \alpha x x^T + A$
	//数学公式：$A \leftarrow \alpha x x^T + A$
	Ssyr(ul Uplo, n int, alpha float32, x []float32, incX int, a []float32, lda int)
	// Sspr - 压缩对称秩1更新
	// 用途：压缩存储的对称矩阵秩1更新
	Sspr(ul Uplo, n int, alpha float32, x []float32, incX int, ap []float32)
	// Ssyr2 - 对称秩2更新
	// 用途：对称矩阵的秩2更新 $A \leftarrow \alpha x y^T + \alpha y x^T + A$
	//数学公式：$A \leftarrow \alpha (x y^T + y x^T) + A$
	Ssyr2(ul Uplo, n int, alpha float32, x []float32, incX int, y []float32, incY int, a []float32, lda int)
	// Sspr2 - 压缩对称秩2更新
	// 用途：压缩存储的对称矩阵秩2更新
	Sspr2(ul Uplo, n int, alpha float32, x []float32, incX int, y []float32, incY int, a []float32)
}

// Float32Level3 implements the single precision real BLAS Level 3 routines.
// Float32Level3 接口提供了单精度浮点数矩阵-矩阵运算的功能，是 BLAS Level 3 操作的单精度版本。在深度学习、计算机图形学、实时信号处理等需要高性能矩阵运算且内存受限的场景中非常有用。
// 深度学习 - 神经网络  全连接层的前向传播（批量处理）; 卷积层通过 im2col + GEMM 实现
// 计算机图形学 -批量变换顶点（模型-视图-投影矩阵）
// 推荐系统 矩阵分解 - 交替最小二乘法
// 实时信号处理  块处理的滤波器组
// 金融风险分析  投资组合协方差矩阵计算
type Float32Level3 interface {
	// Sgemm - 通用矩阵乘法
	// 用途：通用矩阵乘法，深度学习和其他数值计算的核心操作
	//数学公式：$C \leftarrow \alpha op(A) \times op(B) + \beta C$
	//tA, tB：控制是否对 A 或 B 进行转置
	//m, n, k：矩阵维度 (m×k) × (k×n) = (m×n)
	Sgemm(tA, tB Transpose, m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int)
	// Ssymm - 对称矩阵乘法
	// 用途：对称矩阵与一般矩阵的乘法
	//数学公式：
	//side = Left: $C \leftarrow \alpha A B + \beta C$（A 对称）
	//side = Right: $C \leftarrow \alpha B A + \beta C$（A 对称）
	//特性：只需存储对称矩阵的上三角或下三角部分，节省内存
	Ssymm(s Side, ul Uplo, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int)
	// Ssyrk - 对称秩k更新
	// 用途：对称矩阵的秩k更新，用于协方差矩阵计算等
	//数学公式：
	//trans = NoTrans: $C \leftarrow \alpha A A^T + \beta C$
	//trans = Trans: $C \leftarrow \alpha A^T A + \beta C$
	//应用：机器学习中的协方差矩阵、相关矩阵计算
	Ssyrk(ul Uplo, t Transpose, n, k int, alpha float32, a []float32, lda int, beta float32, c []float32, ldc int)
	// Ssyr2k - 对称秩2k更新
	// 用途：对称矩阵的秩2k更新
	//数学公式：
	//trans = NoTrans: $C \leftarrow \alpha A B^T + \alpha B A^T + \beta C$
	//trans = Trans: $C \leftarrow \alpha A^T B + \alpha B^T A + \beta C$
	//应用：更复杂的矩阵更新操作
	Ssyr2k(ul Uplo, t Transpose, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int)
	// Strmm - 三角矩阵乘法
	// 用途：三角矩阵与一般矩阵的乘法
	//数学公式：
	//side = Left: $B \leftarrow \alpha op(A) B$
	//side = Right: $B \leftarrow \alpha B op(A)$
	//参数：
	//diag：Unit（单位三角矩阵）或 NonUnit
	//uplo：Upper（上三角）或 Lower（下三角）
	Strmm(s Side, ul Uplo, tA Transpose, d Diag, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int)
	// Strsm - 三角方程组求解
	// 用途：解三角矩阵方程组（多个右端项）
	//数学公式：
	//side = Left: $op(A) X = \alpha B$，解存储在 B 中
	//side = Right: $X op(A) = \alpha B$，解存储在 B 中
	//应用：线性方程组求解、矩阵分解的后代步骤
	Strsm(s Side, ul Uplo, tA Transpose, d Diag, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int)
}

// Float64 implements the single precision real BLAS routines.
type Float64 interface {
	Float64Level1
	Float64Level2
	Float64Level3
}

// Float64Level1 implements the double precision real BLAS Level 1 routines.
// Float64Level1实现了双精度实数BLAS Level 1例程。 Float64Level1 接口提供了双精度浮点数向量运算的功能，是 BLAS Level 1 操作的双精度版本。在科学计算、工程仿真、金融建模等需要高精度计算的领域中至关重要。
// 科学计算 - 数值积分: 使用梯形法则进行数值积分
// 机器学习 - 梯度下降: 批量梯度下降更新. 计算损失函数的梯度范数（用于收敛检查）
// 金融工程 - 投资组合优化: 计算投资组合收益和风险
// 信号处理 - 滤波器设计: 有限脉冲响应(FIR)滤波器应用
// 计算机图形学 - 几何计算: 3D向量操作
// 数值分析 - 线性代数算法: // Gram-Schmidt正交化
type Float64Level1 interface {
	// Ddot - 向量点积 用途：计算两个向量的点积（内积）
	//数学公式：$\sum_{i=0}^{n-1} x_i \times y_i$
	//应用：相关性计算、投影、能量计算
	Ddot(n int, x []float64, incX int, y []float64, incY int) float64
	// Dnrm2 - 欧几里得范数 用途：计算向量的 L2 范数（欧几里得长度）
	//数学公式：$\sqrt{\sum_{i=0}^{n-1} x_i^2}$
	//应用：向量归一化、距离计算、误差度量
	Dnrm2(n int, x []float64, incX int) float64
	// Dasum - 绝对值求和
	// 用途：计算向量各元素绝对值的和（L1 范数）
	//数学公式：$\sum_{i=0}^{n-1} |x_i|$
	//应用：误差估计、信号能量、优化目标函数
	Dasum(n int, x []float64, incX int) float64
	// Idamax - 最大绝对值索引
	// 用途：找到向量中绝对值最大元素的索引（从0开始）
	//数学公式：$\text{argmax}_i |x_i|$
	//应用：主元选择、特征值分析、峰值检测
	Idamax(n int, x []float64, incX int) int
	// Dswap - 向量交换
	// 用途：交换两个向量的内容
	//数学公式：$x_i \leftrightarrow y_i$
	//应用：算法中的临时交换、内存重排
	Dswap(n int, x []float64, incX int, y []float64, incY int)
	// Dcopy - 向量复制
	// 用途：将向量 x 复制到 y
	//数学公式：$y_i \leftarrow x_i$
	//应用：数据备份、初始化、临时存储
	Dcopy(n int, x []float64, incX int, y []float64, incY int)
	// Daxpy - 标量乘加
	// 用途：计算 $y = \alpha x + y$（AX + Y）
	//数学公式：$y_i \leftarrow \alpha \times x_i + y_i$
	//应用：线性组合、向量更新、数值积分
	Daxpy(n int, alpha float64, x []float64, incX int, y []float64, incY int)
	// Drotg - 构造 Givens 旋转
	// 用途：构造 Givens 旋转矩阵的参数
	//输出：
	//c：余弦值
	//s：正弦值
	//r：$\sqrt{a^2 + b^2}$
	//z：缩放参数
	//应用：QR分解、矩阵对角化、数值稳定性
	Drotg(a, b float64) (c, s, r, z float64)
	// Drotmg - 构造改进的 Givens 旋转
	// 用途：构造改进的 Givens 旋转参数（用于带缩放的旋转）
	//应用：数值稳定的平面旋转
	Drotmg(d1, d2, b1, b2 float64) (p DrotmParams, rd1, rd2, rb1 float64)
	// Drot - 应用 Givens 旋转
	// 用途：对两个向量应用 Givens 旋转
	//数学公式：
	//$\begin{bmatrix} x_i' \ y_i' \end{bmatrix} = \begin{bmatrix} c & s \ -s & c \end{bmatrix} \begin{bmatrix} x_i \ y_i \end{bmatrix}$
	//应用：QR算法、特征值计算
	Drot(n int, x []float64, incX int, y []float64, incY int, c float64, s float64)
	// Drotm - 应用改进的 Givens 旋转
	//  用途：应用改进的 Givens 旋转（带缩放）
	//应用：高性能的平面旋转
	Drotm(n int, x []float64, incX int, y []float64, incY int, p DrotmParams)
	// Dscal - 向量缩放
	// 用途：用标量缩放向量 $x = \alpha x$
	// 数学公式：$x_i \leftarrow \alpha \times x_i$
	// 应用：归一化、单位转换、缩放变换
	Dscal(n int, alpha float64, x []float64, incX int)
}

// Float64Level2 implements the double precision real BLAS Level 2 routines.
// Float64Level2 接口提供了双精度浮点数矩阵-向量运算的功能，是 BLAS Level 2 操作的双精度版本。在科学计算、工程仿真、金融建模等需要高精度矩阵-向量运算的领域中至关重要。
// 科学计算 - 有限元分析; 应用刚度矩阵到位移向量;解线性弹性系统
// 计算流体动力学 应用对流-扩散算子; 更新雅可比矩阵
// 金融工程 - 风险评估  计算投资组合的Delta风险; 更新协方差矩阵
// 机器学习 - 支持向量机  计算核矩阵与向量的乘积;更新对偶变量
// 数值优化 - 拟牛顿法; BFGS 更新
// 控制系统 - 状态估计; 卡尔曼滤波测量更新
type Float64Level2 interface {
	// Dgemv - 通用矩阵-向量乘法
	// 用途：计算通用矩阵与向量的乘积
	//数学公式：$y \leftarrow \alpha op(A) x + \beta y$
	//$op(A) = A$ 或 $A^T$
	//支持非转置、转置操作
	Dgemv(tA Transpose, m, n int, alpha float64, a []float64, lda int, x []float64, incX int, beta float64, y []float64, incY int)
	// Dgbmv - 带状矩阵-向量乘法
	// 用途：带状矩阵与向量的乘积（仅存储非零对角线）
	//参数：
	//kL：下带宽（主对角线以下的非零对角线数量）
	//kU：上带宽（主对角线以上的非零对角线数量）
	Dgbmv(tA Transpose, m, n, kL, kU int, alpha float64, a []float64, lda int, x []float64, incX int, beta float64, y []float64, incY int)
	// Dtrmv - 三角矩阵-向量乘法
	// 用途：三角矩阵与向量的乘积
	//数学公式：$x \leftarrow op(A) x$
	//参数：
	//uplo：Upper 或 Lower（上三角或下三角）
	//diag：Unit 或 NonUnit（单位三角矩阵或普通）
	Dtrmv(ul Uplo, tA Transpose, d Diag, n int, a []float64, lda int, x []float64, incX int)
	// Dtbmv - 三角带状矩阵-向量乘法
	// 用途：三角带状矩阵与向量的乘积
	Dtbmv(ul Uplo, tA Transpose, d Diag, n, k int, a []float64, lda int, x []float64, incX int)
	// Dtpmv - 压缩三角矩阵-向量乘法
	// 用途：压缩存储的三角矩阵与向量的乘积
	Dtpmv(ul Uplo, tA Transpose, d Diag, n int, ap []float64, x []float64, incX int)
	// Dtrsv - 三角方程组求解
	// 用途：解三角方程组 $op(A) x = b$
	// 数学公式：$x \leftarrow A^{-1} x$（原地求解）
	// 应用：线性方程组求解的关键步骤
	Dtrsv(ul Uplo, tA Transpose, d Diag, n int, a []float64, lda int, x []float64, incX int)
	// Dtbsv - 三角带状方程组求解
	// 用途：解三角带状方程组
	Dtbsv(ul Uplo, tA Transpose, d Diag, n, k int, a []float64, lda int, x []float64, incX int)
	// Dtpsv - 压缩三角方程组求解
	// 用途：解压缩存储的三角方程组
	Dtpsv(ul Uplo, tA Transpose, d Diag, n int, ap []float64, x []float64, incX int)
	// Dsymv - 对称矩阵-向量乘法
	// 用途：对称矩阵与向量的乘积
	// 数学公式：$y \leftarrow \alpha A x + \beta y$（A 是对称矩阵）
	// 特性：$A = A^T$，只需存储上三角或下三角
	Dsymv(ul Uplo, n int, alpha float64, a []float64, lda int, x []float64, incX int, beta float64, y []float64, incY int)
	// Dsbmv - 对称带状矩阵-向量乘法
	// 对称带状矩阵与向量的乘积
	Dsbmv(ul Uplo, n, k int, alpha float64, a []float64, lda int, x []float64, incX int, beta float64, y []float64, incY int)
	// Dspmv - 压缩对称矩阵-向量乘法
	// 压缩存储的对称矩阵与向量的乘积
	Dspmv(ul Uplo, n int, alpha float64, ap []float64, x []float64, incX int, beta float64, y []float64, incY int)
	// Dger - 外积（秩1更新）
	// 用途：计算秩1更新 $A \leftarrow \alpha x y^T + A$
	//数学公式：$A_{ij} \leftarrow \alpha x_i y_j + A_{ij}$
	//应用：矩阵的低秩更新，协方差矩阵更新
	Dger(m, n int, alpha float64, x []float64, incX int, y []float64, incY int, a []float64, lda int)
	// Dsyr - 对称秩1更新
	// 用途：对称矩阵的秩1更新 $A \leftarrow \alpha x x^T + A$
	//数学公式：$A \leftarrow \alpha x x^T + A$
	Dsyr(ul Uplo, n int, alpha float64, x []float64, incX int, a []float64, lda int)
	// Dspr - 压缩对称秩1更新
	// 压缩存储的对称矩阵秩1更新
	Dspr(ul Uplo, n int, alpha float64, x []float64, incX int, ap []float64)
	// Dsyr2 - 对称秩2更新
	// 用途：对称矩阵的秩2更新 $A \leftarrow \alpha x y^T + \alpha y x^T + A$
	// 数学公式：$A \leftarrow \alpha (x y^T + y x^T) + A$
	Dsyr2(ul Uplo, n int, alpha float64, x []float64, incX int, y []float64, incY int, a []float64, lda int)
	// Dspr2 - 压缩对称秩2更新
	// 用途：压缩存储的对称矩阵秩2更新
	Dspr2(ul Uplo, n int, alpha float64, x []float64, incX int, y []float64, incY int, a []float64)
}

// Float64Level3 implements the double precision real BLAS Level 3 routines.
// Float64Level3 接口提供了双精度浮点数矩阵-矩阵运算的功能，是 BLAS Level 3 操作的双精度版本。在科学计算、工程仿真、金融建模、机器学习等需要高精度和大规模矩阵运算的领域中至关重要。
// 科学计算 - 有限元分析; 组装全局刚度矩阵; 解线性系统 K * U = F
// 计算流体动力学; 应用雅可比矩阵到多个向量; 更新预处理矩阵
// 金融工程 - 风险分析; 计算投资组合协方差矩阵; 蒙特卡洛模拟中的相关矩阵生成
// 机器学习 - 深度学习; 全连接层的前向传播（批量处理）;卷积层通过 im2col + GEMM 实现
// 数值优化 - 拟牛顿法;  L-BFGS 算法中的紧凑表示更新
// 控制系统 - Riccati 方程求解; 代数Riccati方程求解
type Float64Level3 interface {
	// Dgemm - 通用矩阵乘法
	// 用途：通用矩阵乘法，科学计算和深度学习的核心操作
	//数学公式：$C \leftarrow \alpha op(A) \times op(B) + \beta C$
	//tA, tB：控制是否对 A 或 B 进行转置
	//m, n, k：矩阵维度 (m×k) × (k×n) = (m×n)
	Dgemm(tA, tB Transpose, m, n, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int)
	// Dsymm - 对称矩阵乘法
	// 用途：对称矩阵与一般矩阵的乘法
	//数学公式：
	//side = Left: $C \leftarrow \alpha A B + \beta C$（A 对称）
	//side = Right: $C \leftarrow \alpha B A + \beta C$（A 对称）
	//特性：只需存储对称矩阵的上三角或下三角部分，节省内存和计算量
	Dsymm(s Side, ul Uplo, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int)
	// Dsyrk - 对称秩k更新
	// 用途：对称矩阵的秩k更新，用于协方差矩阵、相关矩阵计算
	//数学公式：
	//trans = NoTrans: $C \leftarrow \alpha A A^T + \beta C$
	//trans = Trans: $C \leftarrow \alpha A^T A + \beta C$
	//应用：机器学习中的协方差矩阵、矩阵的平方、Cholesky分解
	Dsyrk(ul Uplo, t Transpose, n, k int, alpha float64, a []float64, lda int, beta float64, c []float64, ldc int)
	// Dsyr2k - 对称秩2k更新
	// 用途：对称矩阵的秩2k更新
	//数学公式：
	//trans = NoTrans: $C \leftarrow \alpha A B^T + \alpha B A^T + \beta C$
	//trans = Trans: $C \leftarrow \alpha A^T B + \alpha B^T A + \beta C$
	//应用：更复杂的矩阵更新操作，如某些优化算法
	Dsyr2k(ul Uplo, t Transpose, n, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int)
	// Dtrmm - 三角矩阵乘法
	// 用途：三角矩阵与一般矩阵的乘法
	//数学公式：
	//side = Left: $B \leftarrow \alpha op(A) B$
	//side = Right: $B \leftarrow \alpha B op(A)$
	//参数：
	//diag：Unit（单位三角矩阵）或 NonUnit
	//uplo：Upper（上三角）或 Lower（下三角）
	Dtrmm(s Side, ul Uplo, tA Transpose, d Diag, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int)
	// Dtrsm - 三角方程组求解（多右端项）
	// 用途：解三角矩阵方程组（多个右端项）
	//数学公式：
	//side = Left: $op(A) X = \alpha B$，解存储在 B 中
	//side = Right: $X op(A) = \alpha B$，解存储在 B 中
	//应用：线性方程组求解、矩阵分解的后代步骤、矩阵求逆
	Dtrsm(s Side, ul Uplo, tA Transpose, d Diag, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int)
}

// Complex64 implements the single precision complex BLAS routines.
type Complex64 interface {
	Complex64Level1
	Complex64Level2
	Complex64Level3
}

// Complex64Level1 implements the single precision complex BLAS Level 1 routines.
// Complex64Level1 接口提供了单精度复数向量运算的功能，是 BLAS Level 1 操作的单精度复数版本。在信号处理、通信系统、音频处理等需要复数运算且内存受限的场景中非常有用。
// 数字信号处理; 复数FIR滤波器应用;信号能量计算
// 通信系统 - 调制解调;QPSK调制; 相位旋转补偿
// 音频处理 - 频域操作; 频域滤波器应用; 计算频谱幅度
// 雷达信号处理; 脉冲压缩; 多普勒处理
// 图像处理 - 复数变换; 2D DFT的行变换
type Complex64Level1 interface {
	// Cdotu - 常规点积
	// 用途：计算 $\sum x_i y_i$（不取共轭）
	// 数学公式：$\sum_{i=0}^{n-1} x_i \times y_i$
	// 应用：双线性形式、卷积
	Cdotu(n int, x []complex64, incX int, y []complex64, incY int) (dotu complex64)
	// Cdotc - 共轭点积
	// 用途：计算 $\sum x_i^* y_i$（对 x 取共轭）
	// 数学公式：$\sum_{i=0}^{n-1} \overline{x_i} \times y_i$
	// 应用：内积空间、相关性计算
	Cdotc(n int, x []complex64, incX int, y []complex64, incY int) (dotc complex64)
	// Scnrm2 - 复数欧几里得范数
	// 用途：计算复数向量的 L2 范数
	// 数学公式：$\sqrt{\sum_{i=0}^{n-1} |x_i|^2} = \sqrt{\sum_{i=0}^{n-1} (\text{real}(x_i)^2 + \text{imag}(x_i)^2)}$
	// 应用：向量归一化、距离计算
	Scnrm2(n int, x []complex64, incX int) float32
	// Scasum - 复数绝对值求和
	// 用途：计算复数向量各元素绝对值的和
	//数学公式：$\sum_{i=0}^{n-1} |\text{real}(x_i)| + |\text{imag}(x_i)|$
	//应用：信号能量计算、误差估计
	Scasum(n int, x []complex64, incX int) float32
	// Icamax - 最大绝对值索引
	// 用途：找到复数向量中绝对值最大元素的索引
	// 数学公式：$\text{argmax}_i |x_i|$
	// 应用：峰值检测、主元选择
	Icamax(n int, x []complex64, incX int) int
	// Cswap - 复数向量交换
	// 用途：交换两个复数向量的内容
	// 数学公式：$x_i \leftrightarrow y_i$
	// 应用：算法中的临时交换
	Cswap(n int, x []complex64, incX int, y []complex64, incY int)
	// Ccopy - 复数向量复制
	// 用途：将复数向量 x 复制到 y
	// 数学公式：$y_i \leftarrow x_i$
	// 应用：数据备份、初始化
	Ccopy(n int, x []complex64, incX int, y []complex64, incY int)
	// Caxpy - 复数标量乘加
	// 用途：计算 $y = \alpha x + y$
	// 数学公式：$y_i \leftarrow \alpha \times x_i + y_i$
	// 应用：线性组合、滤波器应用
	Caxpy(n int, alpha complex64, x []complex64, incX int, y []complex64, incY int)
	// Cscal - 复数缩放
	// 用途：用复数缩放复数向量 $x = \alpha x$
	// 数学公式：$x_i \leftarrow \alpha \times x_i$（$\alpha$ 为复数）
	// 应用：相位旋转、复数增益
	Cscal(n int, alpha complex64, x []complex64, incX int)
	// Csscal - 实数缩放复数
	// 用途：用实数缩放复数向量 $x = \alpha x$
	// 数学公式：$x_i \leftarrow \alpha \times x_i$（$\alpha$ 为实数）
	// 应用：幅度调整、归一化
	Csscal(n int, alpha float32, x []complex64, incX int)
}

// Complex64Level2 implements the single precision complex BLAS routines Level 2 routines.
// Complex64Level2 接口提供了单精度复数矩阵-向量运算的功能，是 BLAS Level 2 操作的单精度复数版本。在信号处理、通信系统、音频处理等需要复数矩阵运算且内存受限的场景中非常有用。
// 通信系统 - MIMO 信道均衡
// 雷达信号处理 - 波束成形
// 音频处理 - 频域滤波
// 图像处理 - 复数变换
type Complex64Level2 interface {
	// Cgemv - 通用复数矩阵-向量乘法
	// 用途：计算通用复数矩阵与向量的乘积
	// 数学公式：$y \leftarrow \alpha op(A) x + \beta y$
	// $op(A) = A, A^T, \text{或} A^H$（共轭转置）
	Cgemv(tA Transpose, m, n int, alpha complex64, a []complex64, lda int, x []complex64, incX int, beta complex64, y []complex64, incY int)
	// Cgbmv - 复数带状矩阵-向量乘法
	// 用途：复数带状矩阵与向量的乘积（仅存储非零对角线）
	// 参数：kL-下带宽，kU-上带宽
	// 适用：稀疏但结构化的复数矩阵
	Cgbmv(tA Transpose, m, n, kL, kU int, alpha complex64, a []complex64, lda int, x []complex64, incX int, beta complex64, y []complex64, incY int)
	// Ctrmv - 三角矩阵-向量乘法
	// 用途：复数三角矩阵与向量的乘积
	//数学公式：$x \leftarrow op(A) x$
	//应用：前向/后向代入法解三角系统
	Ctrmv(ul Uplo, tA Transpose, d Diag, n int, a []complex64, lda int, x []complex64, incX int)
	// Ctbmv - 三角带状矩阵-向量乘法
	// 用途：复数三角带状矩阵与向量的乘积
	Ctbmv(ul Uplo, tA Transpose, d Diag, n, k int, a []complex64, lda int, x []complex64, incX int)
	// Ctpmv - 压缩三角矩阵-向量乘法
	// 用途：压缩存储的复数三角矩阵与向量的乘积
	Ctpmv(ul Uplo, tA Transpose, d Diag, n int, ap []complex64, x []complex64, incX int)
	// Ctrsv - 三角方程组求解
	// 用途：解复数三角方程组 $op(A) x = b$
	// 数学公式：$x \leftarrow A^{-1} x$（原地求解）
	// 应用：复数线性方程组求解的关键步骤
	Ctrsv(ul Uplo, tA Transpose, d Diag, n int, a []complex64, lda int, x []complex64, incX int)
	// Ctbsv - 三角带状方程组求解
	// 用途：解复数三角带状方程组
	Ctbsv(ul Uplo, tA Transpose, d Diag, n, k int, a []complex64, lda int, x []complex64, incX int)
	// Ctpsv - 压缩三角方程组求解
	// 用途：解压缩存储的复数三角方程组
	Ctpsv(ul Uplo, tA Transpose, d Diag, n int, ap []complex64, x []complex64, incX int)
	// Chemv - 埃尔米特矩阵-向量乘法
	// 用途：埃尔米特矩阵与向量的乘积
	// 数学公式：$y \leftarrow \alpha A x + \beta y$（A 是埃尔米特矩阵）
	// 特性：$A = A^H$（共轭转置等于自身），只需存储上三角或下三角
	Chemv(ul Uplo, n int, alpha complex64, a []complex64, lda int, x []complex64, incX int, beta complex64, y []complex64, incY int)
	// Chbmv - 埃尔米特带状矩阵-向量乘法
	// 用途：埃尔米特带状矩阵与向量的乘积
	Chbmv(ul Uplo, n, k int, alpha complex64, a []complex64, lda int, x []complex64, incX int, beta complex64, y []complex64, incY int)
	// Chpmv - 压缩埃尔米特矩阵-向量乘法
	// 用途：压缩存储的埃尔米特矩阵与向量的乘积
	Chpmv(ul Uplo, n int, alpha complex64, ap []complex64, x []complex64, incX int, beta complex64, y []complex64, incY int)
	// Cgeru - 外积（无共轭）
	// 用途：计算秩1更新 $A \leftarrow \alpha x y^T + A$
	// 数学公式：$A \leftarrow \alpha x y^T + A$（无共轭）
	// 应用：矩阵的低秩更新
	Cgeru(m, n int, alpha complex64, x []complex64, incX int, y []complex64, incY int, a []complex64, lda int)
	// Cgerc - 外积（带共轭）
	// 用途：计算秩1更新 $A \leftarrow \alpha x y^H + A$
	//数学公式：$A \leftarrow \alpha x y^H + A$（对 y 取共轭）
	Cgerc(m, n int, alpha complex64, x []complex64, incX int, y []complex64, incY int, a []complex64, lda int)
	// Cher - 埃尔米特秩1更新
	// 用途：埃尔米特矩阵的秩1更新 $A \leftarrow \alpha x x^H + A$
	//数学公式：$A \leftarrow \alpha x x^H + A$（alpha 为实数）
	Cher(ul Uplo, n int, alpha float32, x []complex64, incX int, a []complex64, lda int)
	// Chpr - 压缩埃尔米特秩1更新
	// 用途：压缩存储的埃尔米特矩阵秩1更新
	Chpr(ul Uplo, n int, alpha float32, x []complex64, incX int, a []complex64)
	// Cher2 - 埃尔米特秩2更新
	// 用途：埃尔米特矩阵的秩2更新 $A \leftarrow \alpha x y^H + \overline{\alpha} y x^H + A$
	Cher2(ul Uplo, n int, alpha complex64, x []complex64, incX int, y []complex64, incY int, a []complex64, lda int)
	// Chpr2 - 压缩埃尔米特秩2更新
	// 用途：压缩存储的埃尔米特矩阵秩2更新
	Chpr2(ul Uplo, n int, alpha complex64, x []complex64, incX int, y []complex64, incY int, ap []complex64)
}

// Complex64Level3 implements the single precision complex BLAS Level 3 routines.
// Complex64Level3 接口提供了单精度复数矩阵-矩阵运算的功能，是 BLAS Level 3 操作的单精度复数版本。在信号处理、通信系统、量子计算模拟等需要高性能复数矩阵运算且内存受限的场景中非常有用。
// 通信系统 - MIMO 信号处理
// 雷达信号处理 - 空时自适应处理
// 量子计算模拟
// 音频/图像处理 - 复数变换
// 金融工程 - 复数风险评估
type Complex64Level3 interface {
	// Cgemm - 通用复数矩阵乘法
	// 用途：通用复数矩阵乘法，复数计算的核心操作
	// 数学公式：$C \leftarrow \alpha op(A) \times op(B) + \beta C$
	// tA, tB：控制是否对 A 或 B 进行转置或共轭转置
	// m, n, k：矩阵维度 (m×k) × (k×n) = (m×n)
	Cgemm(tA, tB Transpose, m, n, k int, alpha complex64, a []complex64, lda int, b []complex64, ldb int, beta complex64, c []complex64, ldc int)
	// Csymm - 对称复数矩阵乘法
	// 用途：对称复数矩阵与一般矩阵的乘法
	// 数学公式：
	// side = Left: $C \leftarrow \alpha A B + \beta C$（A 对称）
	// side = Right: $C \leftarrow \alpha B A + \beta C$（A 对称）
	// 特性：A 是复数对称矩阵（$A = A^T$，但不一定埃尔米特）
	Csymm(s Side, ul Uplo, m, n int, alpha complex64, a []complex64, lda int, b []complex64, ldb int, beta complex64, c []complex64, ldc int)
	// Csyrk - 对称秩k更新
	// 用途：对称复数矩阵的秩k更新
	// 数学公式：
	// trans = NoTrans: $C \leftarrow \alpha A A^T + \beta C$
	// trans = Trans: $C \leftarrow \alpha A^T A + \beta C$
	// 应用：复数协方差矩阵计算
	Csyrk(ul Uplo, t Transpose, n, k int, alpha complex64, a []complex64, lda int, beta complex64, c []complex64, ldc int)
	// Csyr2k - 对称秩2k更新
	// 用途：对称复数矩阵的秩2k更新
	// 数学公式：
	// trans = NoTrans: $C \leftarrow \alpha A B^T + \alpha B A^T + \beta C$
	// trans = Trans: $C \leftarrow \alpha A^T B + \alpha B^T A + \beta C$
	Csyr2k(ul Uplo, t Transpose, n, k int, alpha complex64, a []complex64, lda int, b []complex64, ldb int, beta complex64, c []complex64, ldc int)
	// Ctrmm - 三角矩阵乘法
	// 用途：复数三角矩阵与一般矩阵的乘法
	//数学公式：
	//side = Left: $B \leftarrow \alpha op(A) B$
	//side = Right: $B \leftarrow \alpha B op(A)$
	//应用：解多个复数三角系统、矩阵分解
	Ctrmm(s Side, ul Uplo, tA Transpose, d Diag, m, n int, alpha complex64, a []complex64, lda int, b []complex64, ldb int)
	// Ctrsm - 三角方程组求解
	// 用途：解复数三角矩阵方程组
	//数学公式：
	//side = Left: $op(A) X = \alpha B$，解存储在 B 中
	//side = Right: $X op(A) = \alpha B$，解存储在 B 中
	//应用：复数线性方程组求解、矩阵求逆
	Ctrsm(s Side, ul Uplo, tA Transpose, d Diag, m, n int, alpha complex64, a []complex64, lda int, b []complex64, ldb int)
	// Chemm - 埃尔米特矩阵乘法
	// 用途：埃尔米特矩阵与一般矩阵的乘法
	// 数学公式：
	// side = Left: $C \leftarrow \alpha A B + \beta C$（A 埃尔米特）
	// side = Right: $C \leftarrow \alpha B A + \beta C$（A 埃尔米特）
	// 特性：A 是埃尔米特矩阵（$A = A^H$，对角线为实数）
	Chemm(s Side, ul Uplo, m, n int, alpha complex64, a []complex64, lda int, b []complex64, ldb int, beta complex64, c []complex64, ldc int)
	// Cherk - 埃尔米特秩k更新
	// 用途：埃尔米特矩阵的秩k更新
	// 数学公式：
	// trans = NoTrans: $C \leftarrow \alpha A A^H + \beta C$
	// trans = ConjTrans: $C \leftarrow \alpha A^H A + \beta C$
	// 特性：alpha 和 beta 为实数，结果 C 保持埃尔米特性
	Cherk(ul Uplo, t Transpose, n, k int, alpha float32, a []complex64, lda int, beta float32, c []complex64, ldc int)
	Cher2k(ul Uplo, t Transpose, n, k int, alpha complex64, a []complex64, lda int, b []complex64, ldb int, beta float32, c []complex64, ldc int)
}

// Complex128 implements the double precision complex BLAS routines.
type Complex128 interface {
	Complex128Level1
	Complex128Level2
	Complex128Level3
}

// Complex128Level1 implements the double precision complex BLAS Level 1 routines.
// Complex128Level1 接口提供了复数向量运算的功能，是 BLAS Level 1 操作的复数版本。在处理信号处理、量子力学、电子工程等领域的复数计算时非常有用。
type Complex128Level1 interface {
	// Zdotu - 常规点积. 计算 $\sum x_i y_i$（不取共轭） 数学公式：$\sum_{i=0}^{n-1} x_i \times y_i$
	Zdotu(n int, x []complex128, incX int, y []complex128, incY int) (dotu complex128)
	// Zdotc - 共轭点积  用途：计算 $\sum x_i^* y_i$（对 x 取共轭） 数学公式：$\sum_{i=0}^{n-1} \overline{x_i} \times y_i$
	Zdotc(n int, x []complex128, incX int, y []complex128, incY int) (dotc complex128)
	Dznrm2(n int, x []complex128, incX int) float64
	// Dzasum - 绝对值求和.计算复数向量各元素绝对值的和
	Dzasum(n int, x []complex128, incX int) float64
	// Izamax - 最大绝对值索引 找到复数向量中绝对值最大元素的索引  ：$\text{argmax}_i |x_i|$
	Izamax(n int, x []complex128, incX int) int
	// Zswap - 向量交换 交换两个复数向量的内容. $x_i \leftrightarrow y_i$
	Zswap(n int, x []complex128, incX int, y []complex128, incY int)
	// Zcopy - 向量复制. 用途：将复数向量 x 复制到 y 数学公式：$y_i \leftarrow x_i$
	Zcopy(n int, x []complex128, incX int, y []complex128, incY int)
	// Zaxpy 标量乘加. 计算 $y = \alpha x + y$
	Zaxpy(n int, alpha complex128, x []complex128, incX int, y []complex128, incY int)
	// Zscal - 复数缩放. 用复数缩放复数向量 $x = \alpha x$. 数学公式：$x_i \leftarrow \alpha \times x_i$（$\alpha$ 为复数）
	Zscal(n int, alpha complex128, x []complex128, incX int)
	// Zdscal - 实数缩放. 用实数缩放复数向量 $x = \alpha x$. 数学公式：$x_i \leftarrow \alpha \times x_i$（$\alpha$ 为实数）
	Zdscal(n int, alpha float64, x []complex128, incX int)
}

// Complex128Level2 implements the double precision complex BLAS Level 2 routines.
// Complex128Level2 接口提供了复数矩阵-向量运算的功能，是 BLAS Level 2 操作的复数版本。在处理复数线性系统、量子力学、电磁学等领域的矩阵-向量操作时非常有用。
// // 薛定谔方程求解：Hψ = Eψ
// 麦克斯韦方程组的矩阵求解. A 是复数矩阵（考虑介电常数和导电率）. 使用 Zgemv 进行场计算
// 复数滤波器组应用. 使用 Ztrsv 解三角系统进行递归滤波
type Complex128Level2 interface {
	// Zgemv - 通用矩阵-向量乘法. 用途：计算通用矩阵与向量的乘积
	//数学公式：$y \leftarrow \alpha op(A) x + \beta y$
	//$op(A) = A, A^T, \text{或} A^H$（共轭转置）
	//支持非转置、转置、共轭转置
	Zgemv(tA Transpose, m, n int, alpha complex128, a []complex128, lda int, x []complex128, incX int, beta complex128, y []complex128, incY int)
	// Zgbmv - 带状矩阵-向量乘法
	// 用途：带状矩阵与向量的乘积（仅存储非零对角线）
	//参数：kL-下带宽，kU-上带宽
	//适用：稀疏但结构化的矩阵
	Zgbmv(tA Transpose, m, n int, kL int, kU int, alpha complex128, a []complex128, lda int, x []complex128, incX int, beta complex128, y []complex128, incY int)
	// Ztrmv - 三角矩阵-向量乘法
	// 用途：三角矩阵与向量的乘积
	//数学公式：$x \leftarrow op(A) x$
	//应用：前向/后向代入法解三角系统
	Ztrmv(ul Uplo, tA Transpose, d Diag, n int, a []complex128, lda int, x []complex128, incX int)
	// Ztbmv - 三角带状矩阵-向量乘法
	// 用途：三角带状矩阵与向量的乘积
	Ztbmv(ul Uplo, tA Transpose, d Diag, n, k int, a []complex128, lda int, x []complex128, incX int)
	// Ztpmv - 压缩三角矩阵-向量乘法
	// 用途：压缩存储的三角矩阵与向量的乘积
	Ztpmv(ul Uplo, tA Transpose, d Diag, n int, ap []complex128, x []complex128, incX int)
	// Ztrsv - 三角方程组求解
	// 用途：解三角方程组 $op(A) x = b$
	// 数学公式：$x \leftarrow A^{-1} x$（原地求解）
	// 应用：线性方程组求解的关键步骤
	Ztrsv(ul Uplo, tA Transpose, d Diag, n int, a []complex128, lda int, x []complex128, incX int)
	// Ztbsv - 三角带状方程组求解
	// 解三角带状方程组
	Ztbsv(ul Uplo, tA Transpose, d Diag, n, k int, a []complex128, lda int, x []complex128, incX int)
	// Ztpsv - 压缩三角方程组求解
	// 用途：解压缩存储的三角方程组
	Ztpsv(ul Uplo, tA Transpose, d Diag, n int, ap []complex128, x []complex128, incX int)
	// Zhemv - 埃尔米特矩阵-向量乘法
	// 用途：埃尔米特矩阵与向量的乘积
	//数学公式：$y \leftarrow \alpha A x + \beta y$（A 是埃尔米特矩阵）
	//特性：$A = A^H$（共轭转置等于自身），只需存储上三角或下三角
	Zhemv(ul Uplo, n int, alpha complex128, a []complex128, lda int, x []complex128, incX int, beta complex128, y []complex128, incY int)
	// Zhbmv - 埃尔米特带状矩阵-向量乘法
	// 用途：埃尔米特带状矩阵与向量的乘积
	Zhbmv(ul Uplo, n, k int, alpha complex128, a []complex128, lda int, x []complex128, incX int, beta complex128, y []complex128, incY int)
	// Zhpmv - 埃尔米特压缩矩阵-向量乘法
	// 用途：压缩存储的埃尔米特矩阵与向量的乘积
	Zhpmv(ul Uplo, n int, alpha complex128, ap []complex128, x []complex128, incX int, beta complex128, y []complex128, incY int)
	// Zgeru - 外积（无共轭）
	// 用途：计算秩1更新 $A \leftarrow \alpha x y^T + A$
	//数学公式：$A \leftarrow \alpha x y^T + A$（无共轭）
	//应用：矩阵的低秩更新
	Zgeru(m, n int, alpha complex128, x []complex128, incX int, y []complex128, incY int, a []complex128, lda int)
	// Zgerc - 外积（带共轭）
	// 用途：计算秩1更新 $A \leftarrow \alpha x y^H + A$
	// 数学公式：$A \leftarrow \alpha x y^H + A$（对 y 取共轭）
	Zgerc(m, n int, alpha complex128, x []complex128, incX int, y []complex128, incY int, a []complex128, lda int)
	// Zher - 埃尔米特秩1更新
	// 用途：埃尔米特矩阵的秩1更新 $A \leftarrow \alpha x x^H + A$
	// 数学公式：$A \leftarrow \alpha x x^H + A$（alpha 为实数）
	Zher(ul Uplo, n int, alpha float64, x []complex128, incX int, a []complex128, lda int)
	// Zhpr - 压缩埃尔米特秩1更新
	// 用途：压缩存储的埃尔米特矩阵秩1更新
	Zhpr(ul Uplo, n int, alpha float64, x []complex128, incX int, a []complex128)
	// Zher2 - 埃尔米特秩2更新
	// 用途：埃尔米特矩阵的秩2更新 $A \leftarrow \alpha x y^H + \overline{\alpha} y x^H + A$
	Zher2(ul Uplo, n int, alpha complex128, x []complex128, incX int, y []complex128, incY int, a []complex128, lda int)
	// Zhpr2 - 压缩埃尔米特秩2更新
	// 用途：压缩存储的埃尔米特矩阵秩2更新
	Zhpr2(ul Uplo, n int, alpha complex128, x []complex128, incX int, y []complex128, incY int, ap []complex128)
}

// Complex128Level3 implements the double precision complex BLAS Level 3 routines.
// Complex128Level3 接口提供了复数矩阵-矩阵运算的功能，是 BLAS Level 3 操作的复数版本。在处理大型复数矩阵运算、量子力学模拟、电磁场计算等需要高性能矩阵操作的领域中至关重要。
// 量子力学模拟 // 时间演化算符应用：|ψ(t)⟩ = exp(-iHt/ℏ) |ψ(0)⟩  使用 Zgemm 进行矩阵幂运算或克里福德分解
// 量子电路模拟  应用量子门到多个量子态
// 电磁场计算 频域 Maxwell 方程求解：∇×∇×E - ω²μεE = -iωμJ
// 信号处理 - MIMO 系统. 多输入多输出系统的信道容量计算.  H^H * H 用于容量计算
// 控制系统  Lyapunov 方程求解：A^H X + X A = -Q
type Complex128Level3 interface {
	// Zgemm - 通用矩阵乘法
	// 用途：通用矩阵乘法，BLAS Level 3 的核心操作
	//数学公式：$C \leftarrow \alpha op(A) \times op(B) + \beta C$
	//$op(A) = A, A^T, \text{或} A^H$（共轭转置）
	//支持各种转置和共轭转置组合
	Zgemm(tA, tB Transpose, m, n, k int, alpha complex128, a []complex128, lda int, b []complex128, ldb int, beta complex128, c []complex128, ldc int)
	// Zsymm - 对称矩阵乘法
	// 用途：对称矩阵与矩阵的乘法
	//数学公式：
	//side = Left: $C \leftarrow \alpha A B + \beta C$（A 对称）
	//side = Right: $C \leftarrow \alpha B A + \beta C$（A 对称）
	//特性：A 是复数对称矩阵（$A = A^T$，但不一定埃尔米特）
	Zsymm(s Side, ul Uplo, m, n int, alpha complex128, a []complex128, lda int, b []complex128, ldb int, beta complex128, c []complex128, ldc int)
	// Zsyrk - 对称秩k更新
	// 用途：对称矩阵的秩k更新
	// 数学公式：
	// trans = NoTrans: $C \leftarrow \alpha A A^T + \beta C$
	// trans = Trans: $C \leftarrow \alpha A^T A + \beta C$
	// 应用：协方差矩阵计算、矩阵的平方
	Zsyrk(ul Uplo, t Transpose, n, k int, alpha complex128, a []complex128, lda int, beta complex128, c []complex128, ldc int)
	// Zsyr2k - 对称秩2k更新
	// 用途：对称矩阵的秩2k更新
	// 数学公式：
	// trans = NoTrans: $C \leftarrow \alpha A B^T + \alpha B A^T + \beta C$
	// trans = Trans: $C \leftarrow \alpha A^T B + \alpha B^T A + \beta C$
	Zsyr2k(ul Uplo, t Transpose, n, k int, alpha complex128, a []complex128, lda int, b []complex128, ldb int, beta complex128, c []complex128, ldc int)
	// Ztrmm - 三角矩阵乘法
	// 用途：三角矩阵与矩阵的乘法
	// 数学公式：
	// side = Left: $B \leftarrow \alpha op(A) B$
	// side = Right: $B \leftarrow \alpha B op(A)$
	// 应用：解多个三角系统、矩阵分解
	Ztrmm(s Side, ul Uplo, tA Transpose, d Diag, m, n int, alpha complex128, a []complex128, lda int, b []complex128, ldb int)
	// Ztrsm - 三角方程组求解
	// 用途：解三角矩阵方程组
	// 数学公式：
	// side = Left: $op(A) X = \alpha B$，解存储在 B 中
	// side = Right: $X op(A) = \alpha B$，解存储在 B 中
	// 应用：线性方程组求解、矩阵求逆
	Ztrsm(s Side, ul Uplo, tA Transpose, d Diag, m, n int, alpha complex128, a []complex128, lda int, b []complex128, ldb int)
	// Zhemm - 埃尔米特矩阵乘法
	// 用途：埃尔米特矩阵与矩阵的乘法
	//数学公式：
	//side = Left: $C \leftarrow \alpha A B + \beta C$（A 埃尔米特）
	//side = Right: $C \leftarrow \alpha B A + \beta C$（A 埃尔米特）
	//特性：A 是埃尔米特矩阵（$A = A^H$，对角线为实数）
	Zhemm(s Side, ul Uplo, m, n int, alpha complex128, a []complex128, lda int, b []complex128, ldb int, beta complex128, c []complex128, ldc int)
	// Zherk - 埃尔米特秩k更新
	// 用途：埃尔米特矩阵的秩k更新
	// 数学公式：
	// trans = NoTrans: $C \leftarrow \alpha A A^H + \beta C$
	// trans = ConjTrans: $C \leftarrow \alpha A^H A + \beta C$
	// 特性：alpha 和 beta 为实数，结果 C 保持埃尔米特性
	Zherk(ul Uplo, t Transpose, n, k int, alpha float64, a []complex128, lda int, beta float64, c []complex128, ldc int)
	// Zher2k - 埃尔米特秩2k更新
	// 用途：埃尔米特矩阵的秩2k更新
	// 数学公式：
	// trans = NoTrans: $C \leftarrow \alpha A B^H + \overline{\alpha} B A^H + \beta C$
	// trans = ConjTrans: $C \leftarrow \alpha A^H B + \overline{\alpha} B^H A + \beta C$
	// 特性：beta 为实数，保持结果的埃尔米特性
	Zher2k(ul Uplo, t Transpose, n, k int, alpha complex128, a []complex128, lda int, b []complex128, ldb int, beta float64, c []complex128, ldc int)
}
