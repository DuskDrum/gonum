// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package distuv

// LogProber wraps the LogProb method.
// 概率密度或概率质量的对数值
type LogProber interface {
	// LogProb returns the natural logarithm of the
	// value of the probability density or probability
	// mass function at x.
	LogProb(x float64) float64
}

// Rander wraps the Rand method.
// 从分布中生成随机样本
type Rander interface {
	// Rand returns a random sample drawn from the distribution.
	Rand() float64
}

// RandLogProber is the interface that groups the Rander and LogProber methods.
// 结合 Rander 和 LogProber 功能
type RandLogProber interface {
	Rander
	LogProber
}

// Quantiler wraps the Quantile method.
// 计算分布的分位数（逆累积分布函数）
type Quantiler interface {
	// Quantile returns the minimum value of x from amongst
	// all those values whose CDF value exceeds or equals p.
	Quantile(p float64) float64
}
