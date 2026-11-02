//! Temporal types (M3): using information from the future in a decision does not compile.
//! These are compile-time checks (`Session::check`), so they need no market data.

use kairos::session::Session;

fn check(src: &str) -> Result<(), (String, String)> {
    let mut s = Session::new(Box::new(std::io::sink()));
    s.check("t.krs", src).map_err(|d| (d.code.to_string(), s.render(&d)))
}

fn ok(src: &str) {
    if let Err((_, text)) = check(src) {
        panic!("expected to compile, got:\n{text}");
    }
}

fn fails(src: &str, code: &str, needle: &str) {
    match check(src) {
        Ok(()) => panic!("expected {code}, but it compiled:\n{src}"),
        Err((c, text)) => {
            assert_eq!(c, code, "{text}");
            assert!(text.contains(needle), "expected {needle:?} in:\n{text}");
        }
    }
}

/// The skeleton of a monthly strategy; `$SIGNAL` and `$BODY` are filled by each test.
fn strategy(body: &str) -> String {
    format!(
        "let ret = b3.monthly_returns()
let elig = b3.eligible()
var weights = zeros_like(ret)
for m in 12..size(ret, month) {{
{body}
}}
let ev = b3.evaluate(weights, lag: 0)
"
    )
}

#[test]
fn the_momentum_example_uses_only_the_past() {
    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/momentum.krs")).unwrap();
    ok(&src);
}

#[test]
fn momentum_that_includes_the_current_month_does_not_compile() {
    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/momentum.krs")).unwrap();
    let bad = src.replace("m - 12 ..= m - 2", "m - 12 ..= m");
    assert_ne!(src, bad);
    fails(&bad, "E0301", "known at end(m), but the portfolio of month m is decided at end(m - 1)");
}

#[test]
fn last_month_and_the_universe_are_allowed() {
    ok(&strategy(
        "    let alive = elig[month: m] and not is_nan(ret[month: m - 1])
    weights[month: m] = where(alive, 0.01, 0.0)",
    ));
}

#[test]
fn storing_the_current_month_is_caught() {
    fails(&strategy("    weights[month: m] = fill_nan(ret[month: m], 0.0)"), "E0301", "known at end(m)");
}

#[test]
fn a_branch_on_future_data_is_caught() {
    fails(
        &strategy(
            "    if ret[month: m, asset: 0] > 0.0 {
        weights[month: m, asset: 0] = 1.0
    }",
        ),
        "E0301",
        "use of future information",
    );
}

#[test]
fn skipping_a_month_because_of_future_data_is_caught() {
    fails(
        &strategy(
            "    if is_nan(ret[month: m, asset: 0]) { continue }
    weights[month: m, asset: 0] = 1.0",
        ),
        "E0301",
        "use of future information",
    );
}

#[test]
fn full_sample_statistics_are_caught() {
    // normalizing by the mean of the whole history is a classic look-ahead
    fails(
        &strategy("    weights[month: m] = fill_nan(ret[month: m - 1] - mean(ret, over: month), 0.0)"),
        "E0301",
        "whole history",
    );
}

#[test]
fn values_carried_by_the_loop_are_one_month_older() {
    // `prev` holds the previous iteration's month: known at end(m - 1)
    ok(&strategy(
        "    weights[month: m] = prev
    prev = fill_nan(ret[month: m], 0.0) * 0.0",
    )
    .replace("var weights = zeros_like(ret)", "var weights = zeros_like(ret)\nvar prev = zeros(asset: size(ret, asset))"));
    // ... but one assigned before the store, in the same iteration, is not
    fails(
        &strategy(
            "    cur = fill_nan(ret[month: m], 0.0) * 0.0
    weights[month: m] = cur",
        )
        .replace("var weights = zeros_like(ret)", "var weights = zeros_like(ret)\nvar cur = zeros(asset: size(ret, asset))"),
        "E0301",
        "known at end(m)",
    );
}

#[test]
fn an_index_that_is_not_the_loop_variable_cannot_be_proven() {
    fails(
        &strategy("    weights[month: m] = fill_nan(ret[month: int(float(m) * 1.0)], 0.0)"),
        "E0302",
        "cannot prove",
    );
}

#[test]
fn annotated_functions_promise_when_their_result_is_known() {
    let good = "fn signal(r: Tensor[month, asset; f64], m: i64) -> Tensor[asset; f64] @ end(m - 1) {
    prod(1.0 + fill_nan(r[month: m - 12 ..= m - 2], 0.0), over: month) - 1.0
}
let r = observed(zeros(month: 24, asset: 3))
print(signal(r, 20))
";
    ok(good);
    fails(&good.replace("m - 12 ..= m - 2", "m - 12 ..= m"), "E0301", "known at end(m), but `signal` decides at end(m - 1)");
}

#[test]
fn data_known_later_than_a_function_assumes_is_rejected() {
    let src = "fn last(r: Tensor[month; f64], m: i64) -> f64 @ end(m - 1) {
    r[month: m - 1]
}
let late = observed(zeros(month: 12), lag: 1)
print(last(late, 5))
";
    fails(src, "E0303", "known later than the function assumes");
}

#[test]
fn the_call_of_an_annotated_function_carries_its_time() {
    // signal(r, m) is known at end(m - 1): fine for the weights of month m
    ok(&strategy("    weights[month: m] = where(signal(ret, m) > 0.0, 0.01, 0.0)").replace(
        "let ret = b3.monthly_returns()",
        "fn signal(r: Tensor[month, asset; f64], m: i64) -> Tensor[asset; f64] @ end(m - 1) {
    fill_nan(r[month: m - 1], 0.0)
}
let ret = b3.monthly_returns()",
    ));
    // ... but called with m + 1 it is known at end(m)
    fails(
        &strategy("    weights[month: m] = where(signal(ret, m + 1) > 0.0, 0.01, 0.0)").replace(
            "let ret = b3.monthly_returns()",
            "fn signal(r: Tensor[month, asset; f64], m: i64) -> Tensor[asset; f64] @ end(m - 1) {
    fill_nan(r[month: m - 1], 0.0)
}
let ret = b3.monthly_returns()",
        ),
        "E0301",
        "use of future information",
    );
}

#[test]
fn choosing_a_strategy_by_its_backtest_is_look_ahead() {
    // picking whichever did better over the whole history, then evaluating the pick on that same
    // history, is the self-deception the lab exists to stop
    let src = "let ret = b3.monthly_returns()
var a = zeros_like(ret)
var b = zeros_like(ret)
for m in 12..size(ret, month) {
    a[month: m] = where(fill_nan(ret[month: m - 1], 0.0) > 0.0, 0.01, 0.0)
    b[month: m] = where(fill_nan(ret[month: m - 1], 0.0) < 0.0, 0.01, 0.0)
}
let ea = b3.evaluate(a, lag: 0)
var w = zeros_like(ret)
if ea.sharpe > 0.3 { w = a } else { w = b }
print(b3.evaluate(w, lag: 0).cagr)
";
    fails(src, "E0301", "only at the end of the data");
}

#[test]
fn the_repl_remembers_where_data_came_from() {
    let mut s = Session::new(Box::new(std::io::sink()));
    s.check("1", "let ret = b3.monthly_returns()").unwrap();
    s.check("2", "var w = zeros_like(ret)").unwrap();
    let err = s.check("3", "for m in 1..10 { w[month: m] = fill_nan(ret[month: m], 0.0) }\nb3.evaluate(w, lag: 0)").unwrap_err();
    assert_eq!(err.code, "E0301");
}
