// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package optimize

import (
	"math"
)

// Converger returns the convergence of the optimization based on
// locations found during optimization. Converger must not modify the value of
// the provided Location in any of the methods.
// 用于 判断优化算法是否已经收敛。它提供了统一机制，让不同优化方法能够在迭代过程中检查收敛条件，从而停止迭代，避免不必要的计算
type Converger interface {
	Init(dim int)
	// Converged 判断当前迭代点是否满足收敛条件。
	Converged(loc *Location) Status
}

var (
	_ Converger = NeverTerminate{}
	_ Converger = (*FunctionConverge)(nil)
)

// NeverTerminate implements Converger, always reporting NotTerminated.
// optimize 包下的 NeverTerminate 结构体用于实现 永不收敛策略，在优化过程中始终返回未收敛，适用于测试或特殊迭代控制场景。
type NeverTerminate struct{}

func (NeverTerminate) Init(dim int) {}

func (NeverTerminate) Converged(loc *Location) Status {
	return NotTerminated
}

// FunctionConverge tests for insufficient improvement in the optimum value
// over the last iterations. A FunctionConvergence status is returned if
// there is no significant decrease for FunctionConverge.Iterations. A
// significant decrease is considered if
//
//	f < f_best
//
// and
//
//	f_best - f > FunctionConverge.Relative * maxabs(f, f_best) + FunctionConverge.Absolute
//
// If the decrease is significant, then the iteration counter is reset and
// f_best is updated.
//
// If FunctionConverge.Iterations == 0, it has no effect.
// optimize 包下的 FunctionConverge 结构体用于根据 函数值变化收敛 判定优化是否结束，当连续迭代函数值变化小于设定阈值时认为收敛。
type FunctionConverge struct {
	Absolute   float64
	Relative   float64
	Iterations int

	first bool
	best  float64
	iter  int
}

func (fc *FunctionConverge) Init(dim int) {
	fc.first = true
	fc.best = 0
	fc.iter = 0
}

func (fc *FunctionConverge) Converged(l *Location) Status {
	f := l.F
	if fc.first {
		fc.best = f
		fc.first = false
		return NotTerminated
	}
	if fc.Iterations == 0 {
		return NotTerminated
	}
	maxAbs := math.Max(math.Abs(f), math.Abs(fc.best))
	if f < fc.best && fc.best-f > fc.Relative*maxAbs+fc.Absolute {
		fc.best = f
		fc.iter = 0
		return NotTerminated
	}
	fc.iter++
	if fc.iter < fc.Iterations {
		return NotTerminated
	}
	return FunctionConvergence
}
