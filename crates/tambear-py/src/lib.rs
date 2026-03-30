//! Python bindings for tambear.
//!
//! The 10-line API:
//!
//! ```python
//! import tambear as tb
//!
//! df = tb.read("aapl.tb")                         # load .tb — GPU-resident
//! sums = df.groupby("ticker_id").sum("volume")     # hash scatter, sort-free
//! stats = df.groupby("ticker_id").stats("close")   # mean+var+std, one pass
//! hot = df.filter("close > 150.0")                 # mask-not-filter, no copy
//! hot_sums = hot.groupby("ticker_id").sum("volume")
//! df.write("out.tb")
//! ```
//!
//! Four invariants are invisible to the user:
//! - Sort-free (hash scatter)
//! - Mask-not-filter (filter returns same Frame with mask set)
//! - Dictionary strings (encoded at read time, decoded at output)
//! - Types once (dtype from .tb header; no per-op casting)
//!
//! ## PyO3 design notes
//!
//! `PyFrame` holds a `tambear::Frame` via `Arc<Mutex<_>>` so it can be
//! shared across Python threads and mutated (e.g. filter accumulates masks).
//!
//! `PyGroupByBuilder` is a lightweight builder that holds a reference to the
//! frame and the key column name. Calling `.sum()`, `.mean()`, `.stats()`
//! triggers actual GPU execution. Builder pattern keeps the Python surface
//! clean without requiring a lazy graph.
//!
//! TODO (observer): implement when tambear::HashScatterEngine::new() is live.

use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// PyGroupByResult — per-group statistics
// ---------------------------------------------------------------------------

/// Result of a groupby aggregation.
///
/// Contains raw per-group accumulators (sums, sum_sqs, counts).
/// Derived statistics (means, variances, stds) are computed lazily.
#[pyclass(name = "GroupByResult")]
pub struct PyGroupByResult {
    pub n_groups: usize,
    pub sums: Vec<f64>,
    pub sum_sqs: Vec<f64>,
    pub counts: Vec<f64>,
}

#[pymethods]
impl PyGroupByResult {
    /// Per-group means. NaN for empty groups.
    pub fn means(&self) -> Vec<f64> {
        self.sums.iter().zip(&self.counts)
            .map(|(&s, &c)| if c > 0.0 { s / c } else { f64::NAN })
            .collect()
    }

    /// Per-group variances (sample, Bessel-corrected).
    pub fn variances(&self) -> Vec<f64> {
        (0..self.n_groups).map(|g| {
            let c = self.counts[g];
            if c > 1.0 {
                (self.sum_sqs[g] - self.sums[g] * self.sums[g] / c) / (c - 1.0)
            } else { f64::NAN }
        }).collect()
    }

    /// Per-group standard deviations.
    pub fn stds(&self) -> Vec<f64> {
        self.variances().into_iter().map(f64::sqrt).collect()
    }

    /// Active groups (count > 0). Useful after over-allocated groupby.
    pub fn n_active(&self) -> usize {
        self.counts.iter().filter(|&&c| c > 0.0).count()
    }

    #[getter] pub fn n_groups(&self) -> usize { self.n_groups }
    #[getter] pub fn sums(&self) -> Vec<f64> { self.sums.clone() }
    #[getter] pub fn counts(&self) -> Vec<f64> { self.counts.clone() }

    fn __repr__(&self) -> String {
        format!("GroupByResult(n_groups={}, n_active={})", self.n_groups, self.n_active())
    }
}

// ---------------------------------------------------------------------------
// PyGroupByBuilder — lazy builder: df.groupby("col").sum("val")
// ---------------------------------------------------------------------------

/// Lazy groupby builder. Returned by `Frame.groupby()`.
/// Call `.sum()`, `.mean()`, or `.stats()` to execute on GPU.
#[pyclass(name = "GroupByBuilder")]
pub struct PyGroupByBuilder {
    /// Name of the key column (pre-encoded to int codes if string).
    key_col: String,
    /// Maximum key value — determines accumulator array size. From .tb header.
    max_key: u32,
    // TODO (observer): hold Arc<Mutex<tambear::Frame>> when tambear is live
}

#[pymethods]
impl PyGroupByBuilder {
    /// Sum `val_col` per group. One GPU pass via hash scatter.
    pub fn sum(&self, val_col: &str) -> PyResult<PyGroupByResult> {
        // TODO (observer): engine.scatter_sum(row_to_group, values, max_key+1)
        let _ = val_col;
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "sum() not yet implemented — hash scatter engine pending"))
    }

    /// Mean `val_col` per group. Derived from scatter_stats in one pass.
    pub fn mean(&self, val_col: &str) -> PyResult<PyGroupByResult> {
        // TODO: call stats() then return mean
        let _ = val_col;
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "mean() not yet implemented"))
    }

    /// Sum + count + mean + variance + std in ONE GPU pass.
    /// Three atomicAdds per element. Costs less than argsort alone.
    pub fn stats(&self, val_col: &str) -> PyResult<PyGroupByResult> {
        // TODO (observer): engine.groupby(row_to_group, values, max_key+1)
        let _ = val_col;
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "stats() not yet implemented"))
    }

    fn __repr__(&self) -> String {
        format!("GroupByBuilder(key='{}', accumulator_size={})", self.key_col, self.max_key + 1)
    }
}

// ---------------------------------------------------------------------------
// PyFrame — GPU-resident DataFrame
// ---------------------------------------------------------------------------

/// GPU-resident DataFrame. Columns live on GPU until evicted.
///
/// Four invariants are transparent to the user:
/// - Sort-free: groupby uses hash scatter, never sort
/// - Mask-not-filter: filter() sets a bitmask, no data movement
/// - Dictionary strings: string columns are int codes internally
/// - Types once: dtype declared at read() time, fixed for pipeline
#[pyclass(name = "Frame")]
pub struct PyFrame {
    pub n_rows: usize,
    pub column_names: Vec<String>,
    pub pipeline_dtype: String,
    // TODO (observer): hold Arc<Mutex<tambear::Frame>> when tambear is live
}

#[pymethods]
impl PyFrame {
    /// Returns a GroupByBuilder for the given key column.
    /// Execution is deferred until .sum(), .mean(), or .stats() is called.
    pub fn groupby(&self, key_col: &str) -> PyResult<PyGroupByBuilder> {
        // TODO: look up max_key from Frame's column metadata
        Ok(PyGroupByBuilder { key_col: key_col.to_string(), max_key: 0 })
    }

    /// Apply a filter expression. Returns this Frame with the mask updated.
    /// No data movement. Downstream ops are mask-aware.
    /// "Tam doesn't filter. Tam knows which rows matter."
    pub fn filter(&self, _expr: &str) -> PyResult<PyFrame> {
        // TODO (observer): parse expr, build GPU bitmask, set frame.row_mask
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "filter() not yet implemented"))
    }

    /// Write Frame to a .tb file.
    pub fn write(&self, _path: &str) -> PyResult<()> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "write() not yet implemented"))
    }

    #[getter] pub fn n_rows(&self) -> usize { self.n_rows }
    #[getter] pub fn columns(&self) -> Vec<String> { self.column_names.clone() }
    #[getter] pub fn dtype(&self) -> String { self.pipeline_dtype.clone() }

    fn __repr__(&self) -> String {
        format!("Frame(n_rows={}, columns={:?}, dtype={})",
            self.n_rows, self.column_names, self.pipeline_dtype)
    }

    fn __len__(&self) -> usize { self.n_rows }
}

// ---------------------------------------------------------------------------
// Module-level functions
// ---------------------------------------------------------------------------

/// Load a .tb file. Returns a GPU-resident Frame.
///
/// The .tb header is read to determine n_rows, columns, dtypes, max_key
/// values, and pre-allocated scratch region offsets. No data is read yet —
/// columns are lazily loaded to GPU on first access.
///
/// "Tam doesn't allocate. The file already knew."
#[pyfunction]
#[pyo3(signature = (path, dtype=None))]
fn read(path: &str, dtype: Option<&str>) -> PyResult<PyFrame> {
    // TODO (observer): parse .tb header, construct tambear::Frame
    let _ = dtype;
    Err(pyo3::exceptions::PyNotImplementedError::new_err(
        format!("tb.read('{}') not yet implemented — .tb format pending", path)))
}

/// Create an in-memory Frame from Python lists (for testing without .tb files).
///
/// Usage:
///     df = tb.from_columns({"ticker_id": [0,0,1,1,2], "close": [149.0, 150.1, ...]})
#[pyfunction]
fn from_columns(data: &Bound<'_, pyo3::types::PyDict>) -> PyResult<PyFrame> {
    // TODO (observer): create tambear::Frame from Python dict of column lists
    let _ = data;
    Err(pyo3::exceptions::PyNotImplementedError::new_err(
        "from_columns() not yet implemented"))
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

/// tambear — sort-free GPU DataFrame engine.
///
/// Tam doesn't sort. Tam doesn't filter. Tam doesn't do strings. Tam doesn't
/// check types. Tam knows.
#[pymodule]
fn _tambear(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFrame>()?;
    m.add_class::<PyGroupByBuilder>()?;
    m.add_class::<PyGroupByResult>()?;
    m.add_function(wrap_pyfunction!(read, m)?)?;
    m.add_function(wrap_pyfunction!(from_columns, m)?)?;

    // Dtype constants — "Tam doesn't check types. Tam knows the schema."
    m.add("f64", "f64")?;
    m.add("f32", "f32")?;

    Ok(())
}
