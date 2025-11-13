// Copyright ©2020 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package interp

import "slices"

const (
	differentLengths        = "interp: input slices have different lengths"
	tooFewPoints            = "interp: too few points for interpolation"
	xsNotStrictlyIncreasing = "interp: xs values not strictly increasing"
)

// Predictor predicts the value of a function. It handles both
// interpolation and extrapolation.
// 在 Gonum 的 interp 包中，Predictor 接口定义了插值预测器的核心功能，用于基于已知数据点进行数值预测和插值。
type Predictor interface {
	// Predict returns the predicted value at x.
	// Predict - 基本预测方法. 作用：在给定位置 x 处进行插值预测
	//参数：
	//x：需要预测的坐标位置
	//返回值：在 x 处的预测值
	Predict(x float64) float64
}

// Fitter fits a predictor to data.
// 在 Gonum 的 interp 包中，Fitter 接口定义了插值拟合器的核心功能，用于创建和配置插值器。这个接口是构建各种插值方法的基础。
type Fitter interface {
	// Fit fits a predictor to (X, Y) value pairs provided as two slices.
	// It panics if len(xs) < 2, elements of xs are not strictly increasing
	// or len(xs) != len(ys). Returns an error if fitting fails.
	// Fit - 数据拟合方法
	// 作用：基于给定的数据点 (x, y) 拟合插值函数
	//参数：
	//x：自变量数据点（必须单调递增）
	//y：因变量数据点
	//返回值：错误信息，如果拟合成功返回 nil
	Fit(xs, ys []float64) error
}

// FittablePredictor is a Predictor which can fit itself to data.
// 组合接口
type FittablePredictor interface {
	Fitter
	Predictor
}

// DerivativePredictor predicts both the value and the derivative of
// a function. It handles both interpolation and extrapolation.
// DerivativePredictor预测函数的值和导数。它处理插值和外推。
// 在 Gonum 的 interp 包中，DerivativePredictor 接口扩展了基本的预测功能，增加了导数计算能力。这对于需要分析函数变化率、曲率等微分特性的应用非常有用。
type DerivativePredictor interface {
	Predictor

	// PredictDerivative returns the predicted derivative at x.
	// 并增加了导数预测方法。
	// 作用：计算插值函数在给定点 x 处的一阶导数
	// 参数：
	// x：需要计算导数的坐标位置
	// 返回值：在 x 处的一阶导数值
	PredictDerivative(x float64) float64
}

// Constant predicts a constant value.
type Constant float64

// Predict returns the predicted value at x.
func (c Constant) Predict(x float64) float64 {
	return float64(c)
}

// Function predicts by evaluating itself.
type Function func(float64) float64

// Predict returns the predicted value at x by evaluating fn(x).
func (fn Function) Predict(x float64) float64 {
	return fn(x)
}

// PiecewiseLinear is a piecewise linear 1-dimensional interpolator.
// 在 Gonum 的 interp 包中，PiecewiseLinear 结构体实现了分段线性插值，通过连接相邻数据点的直线段来构建插值函数，是最简单常用的插值方法。
type PiecewiseLinear struct {
	// Interpolated X values.
	xs []float64

	// Interpolated Y data values, same len as ys.
	ys []float64

	// Slopes of Y between neighbouring X values. len(slopes) + 1 == len(xs) == len(ys).
	slopes []float64
}

// Fit fits a predictor to (X, Y) value pairs provided as two slices.
// It panics if len(xs) < 2, elements of xs are not strictly increasing
// or len(xs) != len(ys). Always returns nil.
func (pl *PiecewiseLinear) Fit(xs, ys []float64) error {
	n := len(xs)
	if len(ys) != n {
		panic(differentLengths)
	}
	if n < 2 {
		panic(tooFewPoints)
	}
	pl.slopes = calculateSlopes(xs, ys)
	pl.xs = make([]float64, n)
	pl.ys = make([]float64, n)
	copy(pl.xs, xs)
	copy(pl.ys, ys)
	return nil
}

// Predict returns the interpolation value at x.
func (pl PiecewiseLinear) Predict(x float64) float64 {
	i := findSegment(pl.xs, x)
	if i < 0 {
		return pl.ys[0]
	}
	xI := pl.xs[i]
	if x == xI {
		return pl.ys[i]
	}
	n := len(pl.xs)
	if i == n-1 {
		return pl.ys[n-1]
	}
	return pl.ys[i] + pl.slopes[i]*(x-xI)
}

// PiecewiseConstant is a left-continuous, piecewise constant
// 1-dimensional interpolator.
// 在 Gonum 的 interp 包中，PiecewiseConstant 结构体实现了分段常数（阶梯函数）插值，在每个区间内保持常数值，适用于离散数据或直方图类型的插值。
type PiecewiseConstant struct {
	// Interpolated X values.
	xs []float64

	// Interpolated Y data values, same len as ys.
	ys []float64
}

// Fit fits a predictor to (X, Y) value pairs provided as two slices.
// It panics if len(xs) < 2, elements of xs are not strictly increasing
// or len(xs) != len(ys). Always returns nil.
func (pc *PiecewiseConstant) Fit(xs, ys []float64) error {
	n := len(xs)
	if len(ys) != n {
		panic(differentLengths)
	}
	if n < 2 {
		panic(tooFewPoints)
	}
	for i := 1; i < n; i++ {
		if xs[i] <= xs[i-1] {
			panic(xsNotStrictlyIncreasing)
		}
	}
	pc.xs = make([]float64, n)
	pc.ys = make([]float64, n)
	copy(pc.xs, xs)
	copy(pc.ys, ys)
	return nil
}

// Predict returns the interpolation value at x.
func (pc PiecewiseConstant) Predict(x float64) float64 {
	i := findSegment(pc.xs, x)
	if i < 0 {
		return pc.ys[0]
	}
	if x == pc.xs[i] {
		return pc.ys[i]
	}
	n := len(pc.xs)
	if i == n-1 {
		return pc.ys[n-1]
	}
	return pc.ys[i+1]
}

// findSegment returns 0 <= i < len(xs) such that xs[i] <= x < xs[i + 1], where xs[len(xs)]
// is assumed to be +Inf. If no such i is found, it returns -1.
func findSegment(xs []float64, x float64) int {
	i, found := slices.BinarySearch(xs, x)
	if !found {
		return i - 1
	}
	return i
}

// calculateSlopes calculates slopes (ys[i+1] - ys[i]) / (xs[i+1] - xs[i]).
// It panics if len(xs) < 2, elements of xs are not strictly increasing
// or len(xs) != len(ys).
func calculateSlopes(xs, ys []float64) []float64 {
	n := len(xs)
	if n < 2 {
		panic(tooFewPoints)
	}
	if len(ys) != n {
		panic(differentLengths)
	}
	m := n - 1
	slopes := make([]float64, m)
	prevX := xs[0]
	prevY := ys[0]
	for i := 0; i < m; i++ {
		x := xs[i+1]
		y := ys[i+1]
		dx := x - prevX
		if dx <= 0 {
			panic(xsNotStrictlyIncreasing)
		}
		slopes[i] = (y - prevY) / dx
		prevX = x
		prevY = y
	}
	return slopes
}
