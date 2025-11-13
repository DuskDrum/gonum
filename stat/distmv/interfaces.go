// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package distmv

// 处理 多维概率分布

// Quantiler returns the multi-dimensional inverse cumulative distribution function.
// len(x) must equal len(p), and if x is non-nil, len(x) must also equal len(p).
// If x is nil, a new slice will be allocated and returned, otherwise the quantile
// will be stored in-place into x. All of the values of p must be between 0 and 1,
// or Quantile will panic.
// 提供多维分布的逆累积分布函数（Inverse CDF / Quantile）
// 用于根据概率值生成对应的分布样本（常用于分位数计算或概率映射）
type Quantiler interface {
	Quantile(x, p []float64) []float64
}

// LogProber computes the log of the probability of the point x.
// 计算多维分布在给定点 x 的 对数概率密度
type LogProber interface {
	LogProb(x []float64) float64
}

// Rander generates a random number according to the distribution.
//
// If the input is non-nil, len(x) must equal len(p) and the dimension of the distribution,
// otherwise Quantile will panic.
//
// If the input is nil, a new slice will be allocated and returned.
// 从多维分布生成随机样本
type Rander interface {
	Rand(x []float64) []float64
}

// RandLogProber is both a Rander and a LogProber.
// 结合 Rander 与 LogProber 功能
// 可以既生成随机样本，又计算给定点的对数概率密度
type RandLogProber interface {
	Rander
	LogProber
}
