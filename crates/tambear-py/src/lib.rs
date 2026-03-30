//! Python bindings for tambear.
//!
//! The 10-line API:
//!
//! ```python
//! import tambear as tb
//!
//! df = tb.from_columns({"ticker_id": [0,0,1,1,2], "close": [149.0, 150.1, ...]})
//! sums = df.groupby("ticker_id").sum("close")
//! stats = df.groupby("ticker_id").stats("close")   # mean+var+std, one pass
//! ```
//!
//! Four invariants are invisible to the user:
//! - Sort-free (hash scatter)
//! - Mask-not-filter (filter returns same Frame with mask set)
//! - Dictionary strings (encoded at read time, decoded at output)
//! - Types once (dtype from .tb header; no per-op casting)

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use tambear::HashScatterEngine;

// ---------------------------------------------------------------------------
// Global engine — lazily initialized, shared across all Python calls
// ---------------------------------------------------------------------------

static ENGINE: OnceLock<Arc<Mutex<HashScatterEngine>>> = OnceLock::new();

fn get_engine() -> PyResult<Arc<Mutex<HashScatterEngine>>> {
    let eng = ENGINE.get_or_init(|| {
        Arc::new(Mutex::new(
            HashScatterEngine::new().expect("failed to initialize CUDA scatter engine"),
        ))
    });
    Ok(eng.clone())
}

// ---------------------------------------------------------------------------
// ColumnData — host-side column storage
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum ColumnData {
    I32(Vec<i32>),
    F64(Vec<f64>),
}

impl ColumnData {
    fn as_i32(&self) -> PyResult<&[i32]> {
        match self {
            ColumnData::I32(v) => Ok(v),
            ColumnData::F64(_) => Err(pyo3::exceptions::PyTypeError::new_err(
                "expected integer column for groupby key",
            )),
        }
    }

    fn as_f64(&self) -> PyResult<&[f64]> {
        match self {
            ColumnData::F64(v) => Ok(v),
            ColumnData::I32(v) => {
                // Auto-promote i32 → f64 for value columns
                Err(pyo3::exceptions::PyTypeError::new_err(
                    format!("expected float column for values (got {} ints)", v.len()),
                ))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PyGroupByResult — per-group statistics
// ---------------------------------------------------------------------------

/// Result of a groupby aggregation.
#[pyclass(name = "GroupByResult")]
pub struct PyGroupByResult {
    pub n_groups: usize,
    pub sums: Vec<f64>,
    pub sum_sqs: Vec<f64>,
    pub counts: Vec<f64>,
}

#[pymethods]
impl PyGroupByResult {
    pub fn means(&self) -> Vec<f64> {
        self.sums
            .iter()
            .zip(&self.counts)
            .map(|(&s, &c)| if c > 0.0 { s / c } else { f64::NAN })
            .collect()
    }

    pub fn variances(&self) -> Vec<f64> {
        (0..self.n_groups)
            .map(|g| {
                let c = self.counts[g];
                if c > 1.0 {
                    (self.sum_sqs[g] - self.sums[g] * self.sums[g] / c) / (c - 1.0)
                } else {
                    f64::NAN
                }
            })
            .collect()
    }

    pub fn stds(&self) -> Vec<f64> {
        self.variances().into_iter().map(f64::sqrt).collect()
    }

    pub fn n_active(&self) -> usize {
        self.counts.iter().filter(|&&c| c > 0.0).count()
    }

    #[getter]
    pub fn n_groups(&self) -> usize {
        self.n_groups
    }
    #[getter]
    pub fn get_sums(&self) -> Vec<f64> {
        self.sums.clone()
    }
    #[getter]
    pub fn get_counts(&self) -> Vec<f64> {
        self.counts.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "GroupByResult(n_groups={}, n_active={})",
            self.n_groups,
            self.n_active()
        )
    }
}

// ---------------------------------------------------------------------------
// PyGroupByBuilder — lazy builder: df.groupby("col").sum("val")
// ---------------------------------------------------------------------------

#[pyclass(name = "GroupByBuilder")]
pub struct PyGroupByBuilder {
    key_col: String,
    n_groups: usize,
    columns: Arc<HashMap<String, ColumnData>>,
}

#[pymethods]
impl PyGroupByBuilder {
    /// Sum `val_col` per group. One GPU pass via hash scatter.
    pub fn sum(&self, val_col: &str) -> PyResult<PyGroupByResult> {
        let keys = self.get_keys()?;
        let values = self.get_values(val_col)?;

        let engine = get_engine()?;
        let eng = engine.lock().unwrap();
        let sums = eng
            .scatter_sum(keys, values, self.n_groups)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(PyGroupByResult {
            n_groups: self.n_groups,
            sums,
            sum_sqs: vec![0.0; self.n_groups],
            counts: vec![0.0; self.n_groups], // sum-only, no counts
        })
    }

    /// Mean `val_col` per group. Derived from scatter_stats in one pass.
    pub fn mean(&self, val_col: &str) -> PyResult<PyGroupByResult> {
        self.stats(val_col) // stats() gives everything; mean is derived
    }

    /// Sum + count + mean + variance + std in ONE GPU pass.
    pub fn stats(&self, val_col: &str) -> PyResult<PyGroupByResult> {
        let keys = self.get_keys()?;
        let values = self.get_values(val_col)?;

        let engine = get_engine()?;
        let eng = engine.lock().unwrap();
        let result = eng
            .groupby(keys, values, self.n_groups)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(PyGroupByResult {
            n_groups: result.n_groups,
            sums: result.sums,
            sum_sqs: result.sum_sqs,
            counts: result.counts,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "GroupByBuilder(key='{}', n_groups={})",
            self.key_col, self.n_groups
        )
    }
}

impl PyGroupByBuilder {
    fn get_keys(&self) -> PyResult<&[i32]> {
        self.columns
            .get(&self.key_col)
            .ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!("column '{}' not found", self.key_col))
            })?
            .as_i32()
    }

    fn get_values<'a>(&'a self, val_col: &str) -> PyResult<&'a [f64]> {
        self.columns
            .get(val_col)
            .ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!("column '{}' not found", val_col))
            })?
            .as_f64()
    }
}

// ---------------------------------------------------------------------------
// PyFrame — GPU-resident DataFrame
// ---------------------------------------------------------------------------

#[pyclass(name = "Frame")]
pub struct PyFrame {
    n_rows: usize,
    column_names: Vec<String>,
    columns: Arc<HashMap<String, ColumnData>>,
}

#[pymethods]
impl PyFrame {
    /// Returns a GroupByBuilder for the given key column.
    pub fn groupby(&self, key_col: &str) -> PyResult<PyGroupByBuilder> {
        let col = self
            .columns
            .get(key_col)
            .ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!("column '{}' not found", key_col))
            })?;
        let keys = col.as_i32()?;
        let n_groups = (keys.iter().cloned().max().unwrap_or(-1) + 1) as usize;

        Ok(PyGroupByBuilder {
            key_col: key_col.to_string(),
            n_groups,
            columns: self.columns.clone(),
        })
    }

    #[getter]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }
    #[getter]
    pub fn columns(&self) -> Vec<String> {
        self.column_names.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Frame(n_rows={}, columns={:?})",
            self.n_rows, self.column_names
        )
    }

    fn __len__(&self) -> usize {
        self.n_rows
    }
}

// ---------------------------------------------------------------------------
// Module-level functions
// ---------------------------------------------------------------------------

/// Create a Frame from Python dict of lists.
///
/// Usage:
///     df = tb.from_columns({"ticker_id": [0,0,1,1,2], "close": [149.0, 150.1, ...]})
///
/// Integer lists become i32 key columns. Float lists become f64 value columns.
#[pyfunction]
fn from_columns(data: &Bound<'_, PyDict>) -> PyResult<PyFrame> {
    let mut columns: HashMap<String, ColumnData> = HashMap::new();
    let mut column_names: Vec<String> = Vec::new();
    let mut n_rows: Option<usize> = None;

    for (key, value) in data.iter() {
        let name: String = key.extract()?;
        let list = value.downcast::<PyList>()?;
        let len = list.len();

        // Check consistent length
        if let Some(expected) = n_rows {
            if len != expected {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "column '{}' has {} rows, expected {}",
                    name, len, expected
                )));
            }
        }
        n_rows = Some(len);

        // Try to extract as integers first (for key columns), fall back to float
        let col = if let Ok(vals) = list.extract::<Vec<i32>>() {
            ColumnData::I32(vals)
        } else {
            let vals: Vec<f64> = list.extract()?;
            ColumnData::F64(vals)
        };

        column_names.push(name.clone());
        columns.insert(name, col);
    }

    Ok(PyFrame {
        n_rows: n_rows.unwrap_or(0),
        column_names,
        columns: Arc::new(columns),
    })
}

/// Load a .tb file. Returns a GPU-resident Frame.
#[pyfunction]
#[pyo3(signature = (path, dtype=None))]
fn read(path: &str, dtype: Option<&str>) -> PyResult<PyFrame> {
    let _ = dtype;
    Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
        "tb.read('{}') not yet implemented — .tb format pending",
        path
    )))
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
    Ok(())
}
