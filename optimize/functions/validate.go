// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package functions

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/diff/fd"
	"gonum.org/v1/gonum/floats"
)

// function represents an objective function.
// 在 Gonum 的 optimize/functions 包中，Function 接口是一个抽象接口，用于 统一表示数学优化问题中的目标函数及其导数信息。
// 它提供了一种标准方式，让优化算法能够调用函数值、梯度或 Hessian，而无需关心函数的具体实现细节。
type function interface {
	Func(x []float64) float64
}

// 在 Gonum 的 optimize/gradient 包中，Function 接口是为 梯度计算优化而设计的接口，用于统一表示目标函数及其梯度信息，方便优化算法（如梯度下降、共轭梯度法、拟牛顿法等）使用。
// 它类似于 functions.Function，但更专注于梯度相关操作。
type gradient interface {
	Grad(grad, x []float64) []float64
}

// minimumer is an objective function that can also provide information about
// its minima.
// 在 Gonum 的 optimize/minimumer 包中，
// Function 接口是专门为 **最小化问题（Minimization）**设计的接口，它用于抽象目标函数，统一函数值、梯度和 Hessian 的访问方法，使优化算法可以独立于具体函数实现进行最小化迭代。
type minimumer interface {
	function

	// Minima returns _known_ minima of the function.
	Minima() []Minimum
}

// Minimum represents information about an optimal location of a function.
// 表示优化问题的最优解信息, 通常在函数最小化或优化算法完成后返回，用于描述找到的最优点的位置、函数值以及是否为全局最优。
type Minimum struct {
	// X is the location of the minimum. X may not be nil.
	// 最优点位置
	X []float64
	// F is the value of the objective function at X.
	// 最优点函数值
	F float64
	// Global indicates if the location is a global minimum.
	// 是否为全局最优
	Global bool
}

// 它主要用于 测试和验证优化函数实现的正确性，尤其是在单元测试或基准测试中检查函数值和梯度是否符合预期。下面我详细介绍它的作用和字段：
type funcTest struct {
	// 测试点的输入向量
	X []float64
	// 在 X 处函数的期望值
	// F is the expected function value at X.
	F float64
	// Gradient is the expected gradient at X. If nil, it is not evaluated.
	// 在 X 处函数的期望梯度，如果为 nil 则不检查梯度
	Gradient []float64
}

// TODO(vladimir-ch): Decide and implement an exported testing function:
// func Test(f Function, ??? ) ??? {
// }

const (
	// 默认容差参数
	defaultTol       = 1e-12
	defaultGradTol   = 1e-9
	defaultFDGradTol = 1e-5
)

// testFunction checks that the function can evaluate itself (and its gradient)
// correctly.
func testFunction(f function, ftests []funcTest, t *testing.T) {
	// Make a copy of tests because we may append to the slice.
	tests := make([]funcTest, len(ftests))
	copy(tests, ftests)

	// Get information about the function.
	fMinima, isMinimumer := f.(minimumer)
	fGradient, isGradient := f.(gradient)

	// If the function is a Minimumer, append its minima to the tests.
	if isMinimumer {
		for _, minimum := range fMinima.Minima() {
			// Allocate gradient only if the function can evaluate it.
			var grad []float64
			if isGradient {
				grad = make([]float64, len(minimum.X))
			}
			tests = append(tests, funcTest{
				X:        minimum.X,
				F:        minimum.F,
				Gradient: grad,
			})
		}
	}

	for i, test := range tests {
		F := f.Func(test.X)

		// Check that the function value is as expected.
		if math.Abs(F-test.F) > defaultTol {
			t.Errorf("Test #%d: function value given by Func is incorrect. Want: %v, Got: %v",
				i, test.F, F)
		}

		if test.Gradient == nil {
			continue
		}

		// Evaluate the finite difference gradient.
		fdGrad := fd.Gradient(nil, f.Func, test.X, &fd.Settings{
			Formula: fd.Central,
			Step:    1e-6,
		})

		// Check that the finite difference and expected gradients match.
		if !floats.EqualApprox(fdGrad, test.Gradient, defaultFDGradTol) {
			dist := floats.Distance(fdGrad, test.Gradient, math.Inf(1))
			t.Errorf("Test #%d: numerical and expected gradients do not match. |fdGrad - WantGrad|_∞ = %v",
				i, dist)
		}

		// If the function is a Gradient, check that it computes the gradient correctly.
		if isGradient {
			grad := make([]float64, len(test.Gradient))
			fGradient.Grad(grad, test.X)

			if !floats.EqualApprox(grad, test.Gradient, defaultGradTol) {
				dist := floats.Distance(grad, test.Gradient, math.Inf(1))
				t.Errorf("Test #%d: gradient given by Grad is incorrect. |grad - WantGrad|_∞ = %v",
					i, dist)
			}
		}
	}
}
