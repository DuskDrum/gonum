// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// integrate 包中的 quad/quad.go 文件提供了数值积分（数值求积）的功能，用于计算定积分的数值近似值。这是科学计算中非常重要的基础工具。
// 主要作用是数值积分计算：计算函数在区间上的定积分、处理无法解析积分的函数、提供自适应积分算法
// 自适应求积算法: 自动调整精度、处理函数奇点和不连续点、误差估计和控制
package quad

import (
	"math"
	"sync"
)

// FixedLocationer computes a set of quadrature locations and weights and stores
// them in-place into x and weight respectively. The number of points generated is equal to
// the len(x). The weights and locations should be chosen such that
//
//	int_min^max f(x) dx ≈ \sum_i w_i f(x_i)
//
// 定义了固定节点数值积分规则的核心功能
type FixedLocationer interface {
	// FixedLocations 生成数值积分的节点位置和权重
	//参数：
	//n：节点数量（积分精度）
	//a, b：积分区间 [a, b]
	//
	//返回值：
	//xs：节点位置（求积节点）
	//ws：对应权重（求积权重）
	//
	//数学原理：
	//数值积分公式：$\int_a^b f(x) dx \approx \sum_{i=0}^{n-1} w_i f(x_i)$
	FixedLocations(x, weight []float64, min, max float64)
}

// FixedLocationSingler wraps the FixedLocationSingle method.
// 在 integrate/quad/quad.go 文件中，FixedLocationSingler 接口是单精度固定节点数值积分规则的核心接口。它专门为单精度浮点数计算设计，在内存敏感或性能要求高的场景中特别有用。
type FixedLocationSingler interface {
	// FixedLocationSingle returns the location and weight for
	// element k in a fixed quadrature rule with n total samples
	// and integral bounds from min to max.
	// FixedLocationSingle 作用：生成单精度数值积分的节点位置和权重
	// 参数：
	//n：节点数量（积分精度）
	//a, b：积分区间 [a, b]（单精度）
	//
	//返回值：
	//xs：节点位置数组（单精度）
	//ws：对应权重数组（单精度）
	FixedLocationSingle(n, k int, min, max float64) (x, weight float64)
}

// Fixed approximates the integral of the function f from min to max using a fixed
// n-point quadrature rule. During evaluation, f will be evaluated n times using
// the weights and locations specified by rule. That is, Fixed estimates
//
//	int_min^max f(x) dx ≈ \sum_i w_i f(x_i)
//
// If rule is nil, an acceptable default is chosen, otherwise it is
// assumed that the properties of the integral match the assumptions of rule.
// For example, Legendre assumes that the integration bounds are finite. If
// rule is also a FixedLocationSingler, the quadrature points are computed
// individually rather than as a unit.
//
// If concurrent <= 0, f is evaluated serially, while if concurrent > 0, f
// may be evaluated with at most concurrent simultaneous evaluations.
//
// min must be less than or equal to max, and n must be positive, otherwise
// Fixed will panic.
func Fixed(f func(float64) float64, min, max float64, n int, rule FixedLocationer, concurrent int) float64 {
	// TODO(btracey): When there are Hermite polynomial quadrature, add an additional
	// example to the documentation comment that talks about weight functions.
	if n <= 0 {
		panic("quad: non-positive number of locations")
	}
	if min > max {
		panic("quad: min > max")
	}
	if min == max {
		return 0
	}
	intfunc := f
	// If rule is non-nil it is assumed that the function and the constraints
	// of rule are aligned. If it is nil, wrap the function and do something
	// reasonable.
	// TODO(btracey): Replace wrapping with other quadrature rules when
	// we have rules that support infinite-bound integrals.
	if rule == nil {
		// int_a^b f(x)dx = int_u^-1(a)^u^-1(b) f(u(t))u'(t)dt
		switch {
		case math.IsInf(max, 1) && math.IsInf(min, -1):
			// u(t) = (t/(1-t^2))
			min = -1
			max = 1
			intfunc = func(x float64) float64 {
				v := 1 - x*x
				return f(x/v) * (1 + x*x) / (v * v)
			}
		case math.IsInf(max, 1):
			// u(t) = a + t / (1-t)
			a := min
			min = 0
			max = 1
			intfunc = func(x float64) float64 {
				v := 1 - x
				return f(a+x/v) / (v * v)
			}
		case math.IsInf(min, -1):
			// u(t) = a - (1-t)/t
			a := max
			min = 0
			max = 1
			intfunc = func(x float64) float64 {
				return f(a-(1-x)/x) / (x * x)
			}
		}
		rule = Legendre{}
	}
	singler, isSingler := rule.(FixedLocationSingler)

	var xs, weights []float64
	if !isSingler {
		xs = make([]float64, n)
		weights = make([]float64, n)
		rule.FixedLocations(xs, weights, min, max)
	}

	if concurrent > n {
		concurrent = n
	}

	if concurrent <= 0 {
		var integral float64
		// Evaluate in serial.
		if isSingler {
			for k := 0; k < n; k++ {
				x, weight := singler.FixedLocationSingle(n, k, min, max)
				integral += weight * intfunc(x)
			}
			return integral
		}
		for i, x := range xs {
			integral += weights[i] * intfunc(x)
		}
		return integral
	}

	// Evaluate concurrently
	tasks := make(chan int)

	// Launch distributor
	go func() {
		for i := 0; i < n; i++ {
			tasks <- i
		}
		close(tasks)
	}()

	var mux sync.Mutex
	var integral float64
	var wg sync.WaitGroup
	wg.Add(concurrent)
	for i := 0; i < concurrent; i++ {
		// Launch workers
		go func() {
			defer wg.Done()
			var subIntegral float64
			for k := range tasks {
				var x, weight float64
				if isSingler {
					x, weight = singler.FixedLocationSingle(n, k, min, max)
				} else {
					x = xs[k]
					weight = weights[k]
				}
				f := intfunc(x)
				subIntegral += f * weight
			}
			mux.Lock()
			integral += subIntegral
			mux.Unlock()
		}()
	}
	wg.Wait()
	return integral
}
