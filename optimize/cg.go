// Copyright ©2014 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package optimize

import (
	"math"

	"gonum.org/v1/gonum/floats"
)

const (
	iterationRestartFactor = 6
	angleRestartThreshold  = -0.9
)

var (
	_ Method          = (*CG)(nil)
	_ localMethod     = (*CG)(nil)
	_ NextDirectioner = (*CG)(nil)
)

// CGVariant calculates the scaling parameter, β, used for updating the
// conjugate direction in the nonlinear conjugate gradient (CG) method.
// 非线性共轭梯度法
// 在非线性优化中，共轭梯度法的基本迭代公式为：d_{k+1} = -g_{k+1} + \beta_k d_k
// 其中：
//   - g_k = \nabla f(x_k)：当前梯度
//   - d_k：当前搜索方向
//   - \beta_k：控制“新方向中保留多少旧方向”的权重因子
type CGVariant interface {
	// Init is called at the first iteration and provides a way to initialize
	// any internal state.
	// 初始化方法。 在第一次迭代时调用，用于初始化算法的内部状态
	// 在算法开始时（第 0 次迭代）初始化搜索方向。通常设定为负梯度方向，即： d_0 = -g_0
	Init(loc *Location)
	// Beta returns the value of the scaling parameter that is computed
	// according to the particular variant of the CG method.
	// 计算本次迭代的 β 值（βₖ），决定新方向 d_{k+1} 的更新：d_{k+1} = -g_{k+1} + \beta_k d_k
	Beta(grad, gradPrev, dirPrev []float64) float64
}

var (
	_ CGVariant = (*FletcherReeves)(nil)
	_ CGVariant = (*PolakRibierePolyak)(nil)
	_ CGVariant = (*HestenesStiefel)(nil)
	_ CGVariant = (*DaiYuan)(nil)
	_ CGVariant = (*HagerZhang)(nil)
)

// CG implements the nonlinear conjugate gradient method for solving nonlinear
// unconstrained optimization problems. It is a line search method that
// generates the search directions d_k according to the formula
//
//	d_{k+1} = -∇f_{k+1} + β_k*d_k,   d_0 = -∇f_0.
//
// Variants of the conjugate gradient method differ in the choice of the
// parameter β_k. The conjugate gradient method usually requires fewer function
// evaluations than the gradient descent method and no matrix storage, but
// L-BFGS is usually more efficient.
//
// CG implements a restart strategy that takes the steepest descent direction
// (i.e., d_{k+1} = -∇f_{k+1}) whenever any of the following conditions holds:
//
//   - A certain number of iterations has elapsed without a restart. This number
//     is controllable via IterationRestartFactor and if equal to 0, it is set to
//     a reasonable default based on the problem dimension.
//   - The angle between the gradients at two consecutive iterations ∇f_k and
//     ∇f_{k+1} is too large.
//   - The direction d_{k+1} is not a descent direction.
//   - β_k returned from CGVariant.Beta is equal to zero.
//
// The line search for CG must yield step sizes that satisfy the strong Wolfe
// conditions at every iteration, otherwise the generated search direction
// might fail to be a descent direction. The line search should be more
// stringent compared with those for Newton-like methods, which can be achieved
// by setting the gradient constant in the strong Wolfe conditions to a small
// value.
//
// See also William Hager, Hongchao Zhang, A survey of nonlinear conjugate
// gradient methods. Pacific Journal of Optimization, 2 (2006), pp. 35-58, and
// references therein.
// optimize 包下的 CG 结构体用于实现 非线性共轭梯度法，通过迭代计算搜索方向加速无约束优化问题的收敛。
type CG struct {
	// Linesearcher must satisfy the strong Wolfe conditions at every iteration.
	// If Linesearcher == nil, an appropriate default is chosen.
	Linesearcher Linesearcher
	// Variant implements the particular CG formula for computing β_k.
	// If Variant is nil, an appropriate default is chosen.
	Variant CGVariant
	// InitialStep estimates the initial line search step size, because the CG
	// method does not generate well-scaled search directions.
	// If InitialStep is nil, an appropriate default is chosen.
	InitialStep StepSizer

	// IterationRestartFactor determines the frequency of restarts based on the
	// problem dimension. The negative gradient direction is taken whenever
	// ceil(IterationRestartFactor*(problem dimension)) iterations have elapsed
	// without a restart. For medium and large-scale problems
	// IterationRestartFactor should be set to 1, low-dimensional problems a
	// larger value should be chosen. Note that if the ceil function returns 1,
	// CG will be identical to gradient descent.
	// If IterationRestartFactor is 0, it will be set to 6.
	// CG will panic if IterationRestartFactor is negative.
	IterationRestartFactor float64
	// AngleRestartThreshold sets the threshold angle for restart. The method
	// is restarted if the cosine of the angle between two consecutive
	// gradients is smaller than or equal to AngleRestartThreshold, that is, if
	//  ∇f_k·∇f_{k+1} / (|∇f_k| |∇f_{k+1}|) <= AngleRestartThreshold.
	// A value of AngleRestartThreshold closer to -1 (successive gradients in
	// exact opposite directions) will tend to reduce the number of restarts.
	// If AngleRestartThreshold is 0, it will be set to -0.9.
	// CG will panic if AngleRestartThreshold is not in the interval [-1, 0].
	AngleRestartThreshold float64
	// GradStopThreshold sets the threshold for stopping if the gradient norm
	// gets too small. If GradStopThreshold is 0 it is defaulted to 1e-12, and
	// if it is NaN the setting is not used.
	GradStopThreshold float64

	ls *LinesearchMethod

	status Status
	err    error

	restartAfter    int
	iterFromRestart int

	dirPrev      []float64
	gradPrev     []float64
	gradPrevNorm float64
}

func (cg *CG) Status() (Status, error) {
	return cg.status, cg.err
}

func (*CG) Uses(has Available) (uses Available, err error) {
	return has.gradient()
}

func (cg *CG) Init(dim, tasks int) int {
	cg.status = NotTerminated
	cg.err = nil
	return 1
}

func (cg *CG) Run(operation chan<- Task, result <-chan Task, tasks []Task) {
	cg.status, cg.err = localOptimizer{}.run(cg, cg.GradStopThreshold, operation, result, tasks)
	close(operation)
}

func (cg *CG) initLocal(loc *Location) (Operation, error) {
	if cg.IterationRestartFactor < 0 {
		panic("cg: IterationRestartFactor is negative")
	}
	if cg.AngleRestartThreshold < -1 || cg.AngleRestartThreshold > 0 {
		panic("cg: AngleRestartThreshold not in [-1, 0]")
	}

	if cg.Linesearcher == nil {
		cg.Linesearcher = &MoreThuente{CurvatureFactor: 0.1}
	}
	if cg.Variant == nil {
		cg.Variant = &HestenesStiefel{}
	}
	if cg.InitialStep == nil {
		cg.InitialStep = &FirstOrderStepSize{}
	}

	if cg.IterationRestartFactor == 0 {
		cg.IterationRestartFactor = iterationRestartFactor
	}
	if cg.AngleRestartThreshold == 0 {
		cg.AngleRestartThreshold = angleRestartThreshold
	}

	if cg.ls == nil {
		cg.ls = &LinesearchMethod{}
	}
	cg.ls.Linesearcher = cg.Linesearcher
	cg.ls.NextDirectioner = cg

	return cg.ls.Init(loc)
}

func (cg *CG) iterateLocal(loc *Location) (Operation, error) {
	return cg.ls.Iterate(loc)
}

func (cg *CG) InitDirection(loc *Location, dir []float64) (stepSize float64) {
	dim := len(loc.X)

	cg.restartAfter = int(math.Ceil(cg.IterationRestartFactor * float64(dim)))
	cg.iterFromRestart = 0

	// The initial direction is always the negative gradient.
	copy(dir, loc.Gradient)
	floats.Scale(-1, dir)

	cg.dirPrev = resize(cg.dirPrev, dim)
	copy(cg.dirPrev, dir)
	cg.gradPrev = resize(cg.gradPrev, dim)
	copy(cg.gradPrev, loc.Gradient)
	cg.gradPrevNorm = floats.Norm(loc.Gradient, 2)

	cg.Variant.Init(loc)
	return cg.InitialStep.Init(loc, dir)
}

func (cg *CG) NextDirection(loc *Location, dir []float64) (stepSize float64) {
	copy(dir, loc.Gradient)
	floats.Scale(-1, dir)

	cg.iterFromRestart++
	var restart bool
	if cg.iterFromRestart == cg.restartAfter {
		// Restart because too many iterations have been taken without a restart.
		restart = true
	}

	gDot := floats.Dot(loc.Gradient, cg.gradPrev)
	gNorm := floats.Norm(loc.Gradient, 2)
	if gDot <= cg.AngleRestartThreshold*gNorm*cg.gradPrevNorm {
		// Restart because the angle between the last two gradients is too large.
		restart = true
	}

	// Compute the scaling factor β_k even when restarting, because cg.Variant
	// may be keeping an inner state that needs to be updated at every iteration.
	beta := cg.Variant.Beta(loc.Gradient, cg.gradPrev, cg.dirPrev)
	if beta == 0 {
		// β_k == 0 means that the steepest descent direction will be taken, so
		// indicate that the method is in fact being restarted.
		restart = true
	}
	if !restart {
		// The method is not being restarted, so update the descent direction.
		floats.AddScaled(dir, beta, cg.dirPrev)
		if floats.Dot(loc.Gradient, dir) >= 0 {
			// Restart because the new direction is not a descent direction.
			restart = true
			copy(dir, loc.Gradient)
			floats.Scale(-1, dir)
		}
	}

	// Get the initial line search step size from the StepSizer even if the
	// method was restarted, because StepSizers need to see every iteration.
	stepSize = cg.InitialStep.StepSize(loc, dir)
	if restart {
		// The method was restarted and since the steepest descent direction is
		// not related to the previous direction, discard the estimated step
		// size from cg.InitialStep and use step size of 1 instead.
		stepSize = 1
		// Reset to 0 the counter of iterations taken since the last restart.
		cg.iterFromRestart = 0
	}

	copy(cg.gradPrev, loc.Gradient)
	copy(cg.dirPrev, dir)
	cg.gradPrevNorm = gNorm
	return stepSize
}

func (*CG) needs() struct {
	Gradient bool
	Hessian  bool
} {
	return struct {
		Gradient bool
		Hessian  bool
	}{true, false}
}

// FletcherReeves implements the Fletcher-Reeves variant of the CG method that
// computes the scaling parameter β_k according to the formula
//
//	β_k = |∇f_{k+1}|^2 / |∇f_k|^2.
//
// 共轭梯度法（Conjugate Gradient, CG） 的 Fletcher–Reeves 变体 的结构体。
// FletcherReeves 是 Gonum 共轭梯度法的最基础变体结构体，使用 \beta_k = \frac{g_{k+1}^T g_{k+1}}{g_k^T g_k} 更新方向，是最经典但相对保守的实现。
type FletcherReeves struct {
	prevNorm float64
}

func (fr *FletcherReeves) Init(loc *Location) {
	fr.prevNorm = floats.Norm(loc.Gradient, 2)
}

func (fr *FletcherReeves) Beta(grad, _, _ []float64) (beta float64) {
	norm := floats.Norm(grad, 2)
	beta = (norm / fr.prevNorm) * (norm / fr.prevNorm)
	fr.prevNorm = norm
	return beta
}

// PolakRibierePolyak implements the Polak-Ribiere-Polyak variant of the CG
// method that computes the scaling parameter β_k according to the formula
//
//	β_k = max(0, ∇f_{k+1}·y_k / |∇f_k|^2),
//
// where y_k = ∇f_{k+1} - ∇f_k.
// 非线性共轭梯度法（Conjugate Gradient, CG） 的 Polak–Ribiere–Polyak 变体 的结构体。
// PolakRibierePolyak 是 Gonum 共轭梯度法中常用的 PRP 变体结构体，通过考虑梯度变化来计算 βₖ，使搜索方向更灵活、收敛更快，适合实际非线性优化。
type PolakRibierePolyak struct {
	prevNorm float64
}

func (pr *PolakRibierePolyak) Init(loc *Location) {
	pr.prevNorm = floats.Norm(loc.Gradient, 2)
}

func (pr *PolakRibierePolyak) Beta(grad, gradPrev, _ []float64) (beta float64) {
	norm := floats.Norm(grad, 2)
	dot := floats.Dot(grad, gradPrev)
	beta = (norm*norm - dot) / (pr.prevNorm * pr.prevNorm)
	pr.prevNorm = norm
	return math.Max(0, beta)
}

// HestenesStiefel implements the Hestenes-Stiefel variant of the CG method
// that computes the scaling parameter β_k according to the formula
//
//	β_k = max(0, ∇f_{k+1}·y_k / d_k·y_k),
//
// where y_k = ∇f_{k+1} - ∇f_k.
// 非线性共轭梯度法（Conjugate Gradient, CG） 中 Hestenes–Stiefel (HS) 变体 的结构体
// HestenesStiefel 是 Gonum 共轭梯度法中的 HS 变体结构体，通过对梯度差向量敏感的 β 计算，提高方向调整效率和收敛速度。
type HestenesStiefel struct {
	y []float64
}

func (hs *HestenesStiefel) Init(loc *Location) {
	hs.y = resize(hs.y, len(loc.Gradient))
}

func (hs *HestenesStiefel) Beta(grad, gradPrev, dirPrev []float64) (beta float64) {
	floats.SubTo(hs.y, grad, gradPrev)
	beta = floats.Dot(grad, hs.y) / floats.Dot(dirPrev, hs.y)
	return math.Max(0, beta)
}

// DaiYuan implements the Dai-Yuan variant of the CG method that computes the
// scaling parameter β_k according to the formula
//
//	β_k = |∇f_{k+1}|^2 / d_k·y_k,
//
// where y_k = ∇f_{k+1} - ∇f_k.
// 共轭梯度法（Conjugate Gradient, CG）中 Dai–Yuan 变体
// DaiYuan 是 Gonum 的共轭梯度优化算法中，用于计算方向更新系数 β 的一个稳定变体实现（即 Dai–Yuan 公式的封装）。
type DaiYuan struct {
	y []float64
}

func (dy *DaiYuan) Init(loc *Location) {
	dy.y = resize(dy.y, len(loc.Gradient))
}

func (dy *DaiYuan) Beta(grad, gradPrev, dirPrev []float64) (beta float64) {
	floats.SubTo(dy.y, grad, gradPrev)
	norm := floats.Norm(grad, 2)
	return norm * norm / floats.Dot(dirPrev, dy.y)
}

// HagerZhang implements the Hager-Zhang variant of the CG method that computes the
// scaling parameter β_k according to the formula
//
//	β_k = (y_k - 2 d_k |y_k|^2/(d_k·y_k))·∇f_{k+1} / (d_k·y_k),
//
// where y_k = ∇f_{k+1} - ∇f_k.
// 非线性共轭梯度法（Conjugate Gradient, CG） 中 Hager–Zhang (HZ) 变体 的结构体
// HagerZhang 是 Gonum 共轭梯度算法中最先进、最鲁棒的变体结构体， 通过更复杂的 βₖ 计算公式，在实际非线性优化中表现最稳定、收敛最快，适合工程应用。
type HagerZhang struct {
	y []float64
}

func (hz *HagerZhang) Init(loc *Location) {
	hz.y = resize(hz.y, len(loc.Gradient))
}

func (hz *HagerZhang) Beta(grad, gradPrev, dirPrev []float64) (beta float64) {
	floats.SubTo(hz.y, grad, gradPrev)
	dirDotY := floats.Dot(dirPrev, hz.y)
	gDotY := floats.Dot(grad, hz.y)
	gDotDir := floats.Dot(grad, dirPrev)
	yNorm := floats.Norm(hz.y, 2)
	return (gDotY - 2*gDotDir*yNorm*yNorm/dirDotY) / dirDotY
}
