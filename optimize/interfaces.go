// Copyright ©2014 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package optimize

// A localMethod can optimize an objective function.
//
// It uses a reverse-communication interface between the optimization method
// and the caller. Method acts as a client that asks the caller to perform
// needed operations via Operation returned from Init and Iterate methods.
// This provides independence of the optimization algorithm on user-supplied
// data and their representation, and enables automation of common operations
// like checking for (various types of) convergence and maintaining statistics.
//
// A Method can command an Evaluation, a MajorIteration or NoOperation operations.
//
// An evaluation operation is one or more of the Evaluation operations
// (FuncEvaluation, GradEvaluation, etc.) which can be combined with
// the bitwise or operator. In an evaluation operation, the requested fields of
// Problem will be evaluated at the point specified in Location.X.
// The corresponding fields of Location will be filled with the results that
// can be retrieved upon the next call to Iterate. The Method interface
// requires that entries of Location are not modified aside from the commanded
// evaluations. Thus, the type implementing Method may use multiple Operations
// to set the Location fields at a particular x value.
//
// Instead of an Evaluation, a Method may declare MajorIteration. In
// a MajorIteration, the values in the fields of Location are treated as
// a potential optimizer. The convergence of the optimization routine
// (GradientThreshold, etc.) is checked at this new best point. In
// a MajorIteration, the fields of Location must be valid and consistent.
//
// A Method must not return InitIteration and PostIteration operations. These are
// reserved for the clients to be passed to Recorders. A Method must also not
// combine the Evaluation operations with the Iteration operations.
// 用于封装局部优化算法（Local 优化器，如 BFGS、Newton、共轭梯度等）的核心逻辑。它主要负责 优化迭代的核心流程，包括搜索方向计算、步长更新、状态初始化和迭代终止条件。
type localMethod interface {
	// Init initializes the method based on the initial data in loc, updates it
	// and returns the first operation to be carried out by the caller.
	// The initial location must be valid as specified by Needs.
	// 初始化优化器状态
	initLocal(loc *Location) (Operation, error)

	// Iterate retrieves data from loc, performs one iteration of the method,
	// updates loc and returns the next operation.
	// 执行一次 局部优化迭代。
	iterateLocal(loc *Location) (Operation, error)

	needser
}

// 在 Gonum 的 optimize 包 中，needsER（在源码里通常写作 needsER 或 NeedsER）是一个 内部接口
// 用于描述优化算法对 梯度（Gradient）和 Hessian（或 Hessian 近似）信息的需求。
// 它的作用是让优化器（如 Local）能够 自动判断在迭代过程中是否需要计算梯度和 Hessian，从而统一调度优化算法。
// 用于 描述优化方法对梯度和 Hessian 的需求。它的核心作用是让优化器在迭代过程中 自动判断是否需要计算梯度或 Hessian，从而优化计算效率。
type needser interface {
	// needs specifies information about the objective function needed by the
	// optimizer beyond just the function value. The information is used
	// internally for initialization and must match evaluation types returned
	// by Init and Iterate during the optimization process.
	// needs指定了优化器所需的目标函数的信息，而不仅仅是函数值。
	// 这些信息在内部用于初始化，并且必须与优化过程中Init和Iterate返回的评估类型相匹配。
	needs() struct {
		Gradient bool
		Hessian  bool
	}
}

// Statuser can report the status and any error. It is intended for methods as
// an additional error reporting mechanism apart from the errors returned from
// Init and Iterate.
// 在 Gonum 的 optimize 包 中，Statuser 是一个 接口，用于获取优化器内部的 状态信息，主要用来 观察、监控或调试优化过程。
// 它与优化器的核心迭代逻辑解耦，使得外层可以统一访问算法状态，而不需要直接依赖具体方法（如 BFGS、Newton、CG 等）。
type Statuser interface {
	// Status 返回当前优化器或方法内部的 评估统计信息。
	Status() (Status, error)
}

// Linesearcher is a type that can perform a line search. It tries to find an
// (approximate) minimum of the objective function along the search direction
// dir_k starting at the most recent location x_k, i.e., it tries to minimize
// the function
//
//	φ(step) := f(x_k + step * dir_k) where step > 0.
//
// Typically, a Linesearcher will be used in conjunction with LinesearchMethod
// for performing gradient-based optimization through sequential line searches.
// 在 Gonum 的 optimize 包中，Linesearcher 是一个核心接口，它用于 非线性优化中的步长（Line Search）策略。
// 在局部优化算法（如 BFGS、Newton、共轭梯度法）中，搜索方向 d_k 已经确定，Linesearcher 的作用是 在该方向上选择合适步长 α_k，以保证目标函数充分下降，同时满足收敛条件（如 Wolfe 条件）。
type Linesearcher interface {
	// Init initializes the Linesearcher and a new line search. Value and
	// derivative contain φ(0) and φ'(0), respectively, and step contains the
	// first trial step length. It returns an Operation that must be one of
	// FuncEvaluation, GradEvaluation, FuncEvaluation|GradEvaluation. The
	// caller must evaluate φ(step), φ'(step), or both, respectively, and pass
	// the result to Linesearcher in value and derivative arguments to Iterate.
	// 初始化步长搜索器状态。
	Init(value, derivative float64, step float64) Operation

	// Iterate takes in the values of φ and φ' evaluated at the previous step
	// and returns the next operation.
	//
	// If op is one of FuncEvaluation, GradEvaluation,
	// FuncEvaluation|GradEvaluation, the caller must evaluate φ(step),
	// φ'(step), or both, respectively, and pass the result to Linesearcher in
	// value and derivative arguments on the next call to Iterate.
	//
	// If op is MajorIteration, a sufficiently accurate minimum of φ has been
	// found at the previous step and the line search has concluded. Init must
	// be called again to initialize a new line search.
	//
	// If err is nil, op must not specify another operation. If err is not nil,
	// the values of op and step are undefined.
	// 执行 一次步长迭代，尝试找到满足收敛条件的步长 α_k。
	Iterate(value, derivative float64) (op Operation, step float64, err error)
}

// NextDirectioner implements a strategy for computing a new line search
// direction at each major iteration. Typically, a NextDirectioner will be
// used in conjunction with LinesearchMethod for performing gradient-based
// optimization through sequential line searches.
// 在 Gonum 的 optimize 包 中，NextDirectioner 是一个 内部接口，用于局部优化算法（Local methods）中计算搜索方向。
// 它把搜索方向的计算逻辑从优化循环中解耦出来，使得不同算法（如 BFGS、共轭梯度、Newton）可以统一调用。
// NextDirectioner 通常在每次迭代中由 localMethod 调用，用于生成下一步的搜索方向 d_k。
type NextDirectioner interface {
	// InitDirection initializes the NextDirectioner at the given starting location,
	// putting the initial direction in place into dir, and returning the initial
	// step size. InitDirection must not modify Location.
	InitDirection(loc *Location, dir []float64) (step float64)

	// NextDirection updates the search direction and step size. Location is
	// the location seen at the conclusion of the most recent linesearch. The
	// next search direction is put in place into dir, and the next step size
	// is returned. NextDirection must not modify Location.
	NextDirection(loc *Location, dir []float64) (step float64)
}

// StepSizer can set the next step size of the optimization given the last Location.
// Returned step size must be positive.
// 在 Gonum 的 optimize 包中，StepSizer 是一个核心接口，它用于 确定优化迭代中的初始步长（Step size 或 α₀）。
// 在局部优化算法中，即使搜索方向 d_k 已经确定，步长搜索器通常也需要一个 初始步长猜测，StepSizer 提供了这一机制。
type StepSizer interface {
	Init(loc *Location, dir []float64) float64
	StepSize(loc *Location, dir []float64) float64
}

// A Recorder can record the progress of the optimization, for example to print
// the progress to StdOut or to a log file. A Recorder must not modify any data.
// 在 Gonum 的 optimize 包 中，Recorder 是一个非常重要的接口，用于 记录优化过程中的状态信息。
// 它的核心作用是收集和存储优化过程中的数据，例如迭代次数、函数值、梯度、步长等，以便后续分析、调试、可视化或日志输出。
// 与 Statuser 不同的是，Recorder 不仅提供统计信息，还可以按迭代记录每一步的详细数据。
type Recorder interface {
	Init() error
	Record(*Location, Operation, *Stats) error
}
