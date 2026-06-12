# backtest/report.py
# Generates human-readable + JSON backtest reports with warnings

import json
from dataclasses import asdict
from typing import Union

from backtesting.validators import (
    WalkForwardResult,
    TrainTestResult
)

from backtesting.metrics import (
    PerformanceMetrics
)


# ── Thresholds for interpretation ──────────────────────────────────────────────
THRESHOLDS = {
    "sharpe_good": 1.0,
    "sharpe_great": 1.5,
    "win_rate_min": 45.0,      # below this is concerning
    "max_dd_warn": -20.0,      # beyond this triggers a warning
    "overfit_warn": 2.0,       # train/test return ratio above this = likely overfit
    "consistency_min": 60.0,   # % of profitable folds
    "direction_accuracy_min": 52.0,   # % — above 50% means the model has some edge
}


def _grade_metric(value: float, good: float, great: float, higher_is_better: bool = True) -> str:
    if higher_is_better:
        if value >= great:
            return "✅ Strong"
        elif value >= good:
            return "🟡 Moderate"
        else:
            return "🔴 Weak"
    else:
        if value <= great:
            return "✅ Strong"
        elif value <= good:
            return "🟡 Moderate"
        else:
            return "🔴 Weak"


def _format_metrics(m: PerformanceMetrics) -> str:
    lines = []
    lines.append(f"  Period              : {m.start_date} → {m.end_date} ({m.trading_days} days)")
    lines.append(f"  Total Return        : {m.total_return_pct:+.2f}%")
    lines.append(f"  Benchmark Return    : {m.benchmark_return_pct:+.2f}%  (buy & hold)")
    lines.append(f"  Alpha               : {m.alpha_pct:+.2f}%  {'✅' if m.alpha_pct > 0 else '🔴'}")
    lines.append(f"  Annualized Return   : {m.annualized_return_pct:+.2f}%")
    lines.append(f"  Sharpe Ratio        : {m.sharpe_ratio:.2f}  {_grade_metric(m.sharpe_ratio, 0.8, 1.5)}")
    lines.append(f"  Sortino Ratio       : {m.sortino_ratio:.2f}")
    lines.append(f"  Max Drawdown        : {m.max_drawdown_pct:.2f}%  {'🔴 High' if m.max_drawdown_pct < THRESHOLDS['max_dd_warn'] else '✅ Acceptable'}")
    lines.append(f"  Volatility (Ann.)   : {m.volatility_annualized:.2f}%")
    lines.append(f"  Calmar Ratio        : {m.calmar_ratio:.2f}")
    lines.append(f"  Beta                : {m.beta:.2f}")
    lines.append("")
    lines.append(f"  Total Trades        : {m.total_trades}")
    lines.append(f"  Win Rate            : {m.win_rate_pct:.1f}%  {_grade_metric(m.win_rate_pct, 45, 55)}")
    lines.append(f"  Avg Win             : +{m.avg_win_pct:.2f}%")
    lines.append(f"  Avg Loss            : {m.avg_loss_pct:.2f}%")
    lines.append(f"  Profit Factor       : {m.profit_factor:.2f}  {'✅' if m.profit_factor > 1.2 else '🔴'}")
    if m.prediction_direction_accuracy > 0:
        lines.append(f"  LSTM Direction Acc. : {m.prediction_direction_accuracy:.1f}%  "
                     f"{'✅ Edge' if m.prediction_direction_accuracy > THRESHOLDS['direction_accuracy_min'] else '🔴 No edge over random'}")
        lines.append(f"  LSTM Price MAE      : {m.prediction_mae:.4f}")
        lines.append(f"  BUY Signal Accuracy : {m.signal_accuracy_pct:.1f}%")
    return "\n".join(lines)


def _generate_warnings(result: Union[WalkForwardResult, TrainTestResult]) -> list:
    warnings = []

    if isinstance(result, TrainTestResult):
        tm = result.test_metrics
        if result.overfit_score > THRESHOLDS["overfit_warn"]:
            warnings.append(
                f"⚠️  OVERFITTING DETECTED: Train return ({result.train_metrics.total_return_pct:.1f}%) "
                f"is {result.overfit_score:.1f}x the test return ({tm.total_return_pct:.1f}%). "
                f"Your model may be memorising training data, not learning patterns."
            )
        if tm.prediction_direction_accuracy > 0 and tm.prediction_direction_accuracy < THRESHOLDS["direction_accuracy_min"]:
            warnings.append(
                f"⚠️  LSTM HAS NO DIRECTIONAL EDGE: Direction accuracy is {tm.prediction_direction_accuracy:.1f}% "
                f"(≤50% = worse than a coin flip on test data). Consider a different architecture or features."
            )
        if tm.sharpe_ratio < 0:
            warnings.append("🔴 NEGATIVE SHARPE on test data. Strategy destroys value after accounting for risk.")
        if tm.max_drawdown_pct < THRESHOLDS["max_dd_warn"]:
            warnings.append(f"⚠️  HIGH DRAWDOWN: {tm.max_drawdown_pct:.1f}%. Users would likely abandon the strategy mid-drawdown.")
        if tm.total_trades < 10:
            warnings.append(f"⚠️  TOO FEW TRADES ({tm.total_trades}): Results are not statistically significant.")

    elif isinstance(result, WalkForwardResult):
        if result.consistency_score < THRESHOLDS["consistency_min"]:
            warnings.append(
                f"⚠️  LOW CONSISTENCY: Only {result.consistency_score:.0f}% of folds were profitable. "
                f"The strategy does not work reliably across different market periods."
            )
        if result.avg_sharpe < 0.5:
            warnings.append(f"⚠️  LOW AVERAGE SHARPE ({result.avg_sharpe:.2f}) across folds. "
                            f"Strategy may not have meaningful risk-adjusted edge.")
        if result.avg_prediction_accuracy < THRESHOLDS["direction_accuracy_min"]:
            warnings.append(
                f"⚠️  LSTM DIRECTION ACCURACY ({result.avg_prediction_accuracy:.1f}%) is near random. "
                f"The prediction signal may not be adding value."
            )

    if not warnings:
        warnings.append("✅ No major red flags detected. Review individual fold metrics before deploying.")

    return warnings


def generate_report(
    result: Union[WalkForwardResult, TrainTestResult],
    output_path: str = None,
) -> str:
    lines = []
    sep = "─" * 60

    if isinstance(result, TrainTestResult):
        lines.append(f"\n{'═'*60}")
        lines.append(f"  BACKTEST REPORT — {result.ticker}  (Train/Test Split)")
        lines.append(f"{'═'*60}\n")

        lines.append("TRAIN SET")
        lines.append(sep)
        lines.append(_format_metrics(result.train_metrics))

        lines.append(f"\nTEST SET  ← this is what matters")
        lines.append(sep)
        lines.append(_format_metrics(result.test_metrics))

        lines.append(f"\nOVERFIT SCORE: {result.overfit_score:.2f}x  "
                     f"{'🔴 Likely overfit' if result.overfit_score > THRESHOLDS['overfit_warn'] else '✅ Acceptable'}")

    elif isinstance(result, WalkForwardResult):
        lines.append(f"\n{'═'*60}")
        lines.append(f"  WALK-FORWARD REPORT — {result.ticker}")
        lines.append(f"{'═'*60}\n")

        lines.append("FOLD SUMMARY")
        lines.append(sep)
        for fold in result.folds:
            m = fold.metrics
            lines.append(
                f"  Fold {fold.fold_number}  [{fold.test_start} → {fold.test_end}]  "
                f"Return: {m.total_return_pct:+.1f}%  Sharpe: {m.sharpe_ratio:.2f}  "
                f"WinRate: {m.win_rate_pct:.0f}%  Trades: {fold.n_trades}"
            )

        lines.append(f"\nAGGREGATED RESULTS")
        lines.append(sep)
        lines.append(f"  Avg Return          : {result.avg_total_return:+.2f}%")
        lines.append(f"  Avg Sharpe          : {result.avg_sharpe:.2f}  {_grade_metric(result.avg_sharpe, 0.8, 1.5)}")
        lines.append(f"  Avg Win Rate        : {result.avg_win_rate:.1f}%")
        lines.append(f"  Avg Max Drawdown    : {result.avg_max_drawdown:.2f}%")
        lines.append(f"  LSTM Dir. Accuracy  : {result.avg_prediction_accuracy:.1f}%")
        lines.append(f"  Consistency Score   : {result.consistency_score:.0f}%  "
                     f"(% of profitable folds)  "
                     f"{'✅' if result.consistency_score >= THRESHOLDS['consistency_min'] else '🔴'}")

        if result.folds:
            lines.append(f"\nBEST FOLD:  #{max(result.folds, key=lambda f: f.metrics.total_return_pct).fold_number}")
            lines.append(f"WORST FOLD: #{min(result.folds, key=lambda f: f.metrics.total_return_pct).fold_number}")

    lines.append(f"\nWARNINGS & INSIGHTS")
    lines.append(sep)
    for w in _generate_warnings(result):
        lines.append(f"  {w}")

    lines.append(f"\n{'═'*60}\n")

    report_str = "\n".join(lines)
    print(report_str)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report_str)
        # Also write JSON
        json_path = output_path.replace(".txt", ".json")
        with open(json_path, "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)
        print(f"Report saved to {output_path}")

    return report_str
