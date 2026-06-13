use ndarray::{Array1, Array2};
use sprs::{CsMat, TriMat};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

use crate::data::base::{Dataset, DatasetValue};
use crate::utils::format::directed2undirected;

/// Errors raised by graph parsing functions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// Edge array does not have exactly two columns.
    InvalidEdgeArrayShape,
    /// Adjacency list could not be converted to edges.
    InvalidAdjacencyList,
    /// File I/O failed while reading input.
    CsvIoError,
    /// Input file format or bounds check failed.
    CsvFormatError,
    /// GraphML parsing failed.
    GraphMlError,
}

fn ensure_in_bounds(i: usize, j: usize, shape: (usize, usize)) -> Result<(), ParseError> {
    if i >= shape.0 || j >= shape.1 {
        Err(ParseError::CsvFormatError)
    } else {
        Ok(())
    }
}

/// Parsed graph bundle with matrices and optional metadata.
#[derive(Debug, Clone)]
pub struct GraphDataset {
    /// Square adjacency matrix for unipartite graphs.
    pub adjacency: Option<CsMat<f64>>,
    /// Rectangular biadjacency matrix for bipartite graphs.
    pub biadjacency: Option<CsMat<f64>>,
    /// Numeric node identifiers after reindexing.
    pub names: Option<Vec<i64>>,
    /// String node identifiers for labeled graphs.
    pub names_str: Option<Vec<String>>,
    /// Numeric row identifiers for bipartite graphs.
    pub names_row: Option<Vec<i64>>,
    /// Numeric column identifiers for bipartite graphs.
    pub names_col: Option<Vec<i64>>,
    /// Per-node attribute descriptions.
    pub node_attribute: Option<Dataset>,
    /// Per-edge attribute descriptions.
    pub edge_attribute: Option<Dataset>,
    /// File-level metadata.
    pub meta: Option<Dataset>,
}

/// Result of a parse operation: matrix only or full dataset.
#[derive(Debug, Clone)]
pub enum ParseResult {
    /// Sparse adjacency or biadjacency matrix.
    Matrix(CsMat<f64>),
    /// Full graph dataset with optional sidecars.
    Dataset(GraphDataset),
}

/// Builds a sparse matrix from a two-column edge array.
///
/// # Arguments
/// - `edge_array`: `n_edges × 2` endpoint array.
/// - `weights`: Optional per-edge weights.
/// - `directed`: Keep edge direction when false symmetrizes.
/// - `bipartite`: Build a biadjacency matrix when true.
/// - `weighted`: Use provided weights when true.
/// - `reindex`: Remap node ids to contiguous indices when true.
/// - `shape`: Optional output matrix shape.
/// - `matrix_only`: Return [`ParseResult::Matrix`] when true.
///
/// # Errors
/// Returns [`ParseError`] variants for shape, bounds, or format failures.
pub fn from_edge_array(
    edge_array: &Array2<i64>,
    weights: Option<&Array1<f64>>,
    directed: bool,
    bipartite: bool,
    weighted: bool,
    reindex: bool,
    shape: Option<(usize, usize)>,
    matrix_only: Option<bool>,
) -> Result<ParseResult, ParseError> {
    if edge_array.ncols() != 2 {
        return Err(ParseError::InvalidEdgeArrayShape);
    }
    let n_edges = edge_array.nrows();
    let mut vals = if let Some(w) = weights {
        w.to_vec()
    } else {
        vec![1.0; n_edges]
    };
    if !weighted {
        vals.fill(1.0);
    }

    if bipartite {
        let mut rows: Vec<i64> = edge_array.column(0).to_vec();
        let mut cols: Vec<i64> = edge_array.column(1).to_vec();
        if rows.iter().any(|&x| x < 0) || cols.iter().any(|&x| x < 0) {
            return Err(ParseError::CsvFormatError);
        }
        let mut names_row = None;
        let mut names_col = None;
        if reindex {
            let mut ur = rows.clone();
            ur.sort_unstable();
            ur.dedup();
            let mut uc = cols.clone();
            uc.sort_unstable();
            uc.dedup();
            let map_r = ur
                .iter()
                .enumerate()
                .map(|(i, &v)| (v, i))
                .collect::<std::collections::HashMap<_, _>>();
            let map_c = uc
                .iter()
                .enumerate()
                .map(|(i, &v)| (v, i))
                .collect::<std::collections::HashMap<_, _>>();
            for r in &mut rows {
                *r = *map_r.get(r).unwrap_or(&0) as i64;
            }
            for c in &mut cols {
                *c = *map_c.get(c).unwrap_or(&0) as i64;
            }
            names_row = Some(ur);
            names_col = Some(uc);
        }
        let n_row = shape
            .map(|s| s.0)
            .unwrap_or_else(|| rows.iter().copied().max().unwrap_or(0) as usize + 1);
        let n_col = shape
            .map(|s| s.1)
            .unwrap_or_else(|| cols.iter().copied().max().unwrap_or(0) as usize + 1);
        let mut tri = TriMat::<f64>::new((n_row, n_col));
        for i in 0..n_edges {
            let ri = rows[i] as usize;
            let ci = cols[i] as usize;
            ensure_in_bounds(ri, ci, (n_row, n_col))?;
            tri.add_triplet(ri, ci, vals[i]);
        }
        let mat = tri.to_csr::<usize>();
        if matrix_only.unwrap_or(true) {
            Ok(ParseResult::Matrix(mat))
        } else {
            Ok(ParseResult::Dataset(GraphDataset {
                adjacency: None,
                biadjacency: Some(mat),
                names: None,
                names_str: None,
                names_row,
                names_col,
                node_attribute: None,
                edge_attribute: None,
                meta: None,
            }))
        }
    } else {
        let mut nodes: Vec<i64> = edge_array.iter().copied().collect();
        let mut row = edge_array.column(0).to_vec();
        let mut col = edge_array.column(1).to_vec();
        if row.iter().any(|&x| x < 0) || col.iter().any(|&x| x < 0) {
            return Err(ParseError::CsvFormatError);
        }
        let mut names = None;
        if reindex {
            nodes.sort_unstable();
            nodes.dedup();
            let map = nodes
                .iter()
                .enumerate()
                .map(|(i, &v)| (v, i))
                .collect::<std::collections::HashMap<_, _>>();
            for r in &mut row {
                *r = *map.get(r).unwrap_or(&0) as i64;
            }
            for c in &mut col {
                *c = *map.get(c).unwrap_or(&0) as i64;
            }
            names = Some(nodes);
        }
        let n = shape
            .map(|s| s.0.max(s.1))
            .unwrap_or_else(|| row.iter().chain(col.iter()).copied().max().unwrap_or(0) as usize + 1);
        let mut tri = TriMat::<f64>::new((n, n));
        for i in 0..n_edges {
            let ri = row[i] as usize;
            let ci = col[i] as usize;
            ensure_in_bounds(ri, ci, (n, n))?;
            tri.add_triplet(ri, ci, vals[i]);
        }
        let mut mat = tri.to_csr::<usize>();
        if !directed {
            mat = directed2undirected(&mat, weighted);
        }
        if matrix_only.unwrap_or(names.is_none()) {
            Ok(ParseResult::Matrix(mat))
        } else {
            Ok(ParseResult::Dataset(GraphDataset {
                adjacency: Some(mat),
                biadjacency: None,
                names,
                names_str: None,
                names_row: None,
                names_col: None,
                node_attribute: None,
                edge_attribute: None,
                meta: None,
            }))
        }
    }
}

/// Builds a sparse matrix from an edge list.
///
/// # Arguments
/// - `edge_list`: `(source, target, optional weight)` triples.
/// - `directed`, `bipartite`, `weighted`, `reindex`, `shape`, `matrix_only`:
///   Same semantics as [`from_edge_array`].
///
/// # Errors
/// Returns the same errors as [`from_edge_array`].
pub fn from_edge_list(
    edge_list: &[(i64, i64, Option<f64>)],
    directed: bool,
    bipartite: bool,
    weighted: bool,
    reindex: bool,
    shape: Option<(usize, usize)>,
    matrix_only: Option<bool>,
) -> Result<ParseResult, ParseError> {
    let mut edge_array = Array2::<i64>::zeros((edge_list.len(), 2));
    let mut w = Array1::<f64>::zeros(edge_list.len());
    for (i, (u, v, wt)) in edge_list.iter().enumerate() {
        edge_array[[i, 0]] = *u;
        edge_array[[i, 1]] = *v;
        w[i] = wt.unwrap_or(1.0);
    }
    from_edge_array(
        &edge_array,
        Some(&w),
        directed,
        bipartite,
        weighted,
        reindex,
        shape,
        matrix_only,
    )
}

/// Builds a sparse matrix from an adjacency list.
///
/// # Arguments
/// - `adjacency_list`: Neighbor index lists per source node.
/// - Remaining flags match [`from_edge_list`].
///
/// # Errors
/// Returns the same errors as [`from_edge_list`].
pub fn from_adjacency_list(
    adjacency_list: &[Vec<i64>],
    directed: bool,
    bipartite: bool,
    weighted: bool,
    reindex: bool,
    shape: Option<(usize, usize)>,
    matrix_only: Option<bool>,
) -> Result<ParseResult, ParseError> {
    let mut edges = Vec::<(i64, i64, Option<f64>)>::new();
    for (i, neigh) in adjacency_list.iter().enumerate() {
        for &j in neigh {
            edges.push((i as i64, j, None));
        }
    }
    from_edge_list(
        &edges,
        directed,
        bipartite,
        weighted,
        reindex,
        shape,
        matrix_only,
    )
}

/// Returns whether a string parses as a floating-point number.
pub fn is_number<S: AsRef<str>>(s: S) -> bool {
    s.as_ref().parse::<f64>().is_ok()
}

/// Scans a text file header to guess delimiter and data layout.
///
/// # Arguments
/// - `file_path`: Path to the input file.
/// - `delimiters`: Candidate delimiter characters (`None` uses tab/comma/semicolon/space).
/// - `comments`: Comment-prefix characters to skip.
/// - `n_scan`: Maximum data rows to sample.
///
/// # Errors
/// Returns [`ParseError::CsvIoError`] or [`ParseError::CsvFormatError`].
pub fn scan_header(
    file_path: &str,
    delimiters: Option<&str>,
    comments: &str,
    n_scan: usize,
) -> Result<(usize, char, char, String), ParseError> {
    let file = File::open(file_path).map_err(|_| ParseError::CsvIoError)?;
    let reader = BufReader::new(file);
    let delim_chars: Vec<char> = delimiters.unwrap_or("\t,; ").chars().collect();
    let comment_chars: Vec<char> = comments.chars().collect();
    let mut header_length = 0usize;
    let mut comment_guess = comment_chars.first().copied().unwrap_or('#');
    let mut rows: Vec<String> = Vec::new();
    let mut counts: Vec<Vec<usize>> = vec![Vec::new(); delim_chars.len()];

    for line_res in reader.lines() {
        let line = line_res.map_err(|_| ParseError::CsvIoError)?;
        if line.is_empty()
            || line
                .chars()
                .next()
                .map(|c| comment_chars.contains(&c))
                .unwrap_or(false)
        {
            if let Some(c) = line.chars().next() {
                comment_guess = c;
            }
            header_length += 1;
            continue;
        }
        rows.push(line.clone());
        for (k, d) in delim_chars.iter().enumerate() {
            counts[k].push(line.chars().filter(|c| c == d).count());
        }
        if rows.len() == n_scan {
            break;
        }
    }
    if rows.is_empty() {
        return Err(ParseError::CsvFormatError);
    }
    let mut best_idx = 0usize;
    let mut best_score = f64::MIN;
    for (i, c) in counts.iter().enumerate() {
        let mean = if c.is_empty() {
            0.0
        } else {
            c.iter().sum::<usize>() as f64 / c.len() as f64
        };
        if mean > best_score {
            best_score = mean;
            best_idx = i;
        }
    }
    let delimiter_guess = delim_chars[best_idx];
    let lengths: std::collections::HashSet<usize> = rows
        .iter()
        .map(|r| r.split(delimiter_guess).filter(|s| !s.is_empty()).count())
        .collect();
    let data_structure_guess = if lengths == [2usize].into_iter().collect()
        || lengths == [3usize].into_iter().collect()
    {
        "edge_list".to_string()
    } else {
        "adjacency_list".to_string()
    };
    Ok((
        header_length,
        delimiter_guess,
        comment_guess,
        data_structure_guess,
    ))
}

/// Parses a delimiter-separated graph file into a matrix or dataset.
///
/// # Arguments
/// - `file_path`: Path to the input file.
/// - `delimiter` / `sep`: Explicit delimiter overrides.
/// - `comments`: Comment-prefix characters.
/// - `data_structure`: `edge_list` or `adjacency_list` override.
/// - `directed`, `bipartite`, `weighted`, `reindex`, `shape`, `matrix_only`:
///   Same semantics as [`from_edge_list`].
///
/// # Errors
/// Returns [`ParseError`] variants for I/O, format, or bounds failures.
#[allow(clippy::too_many_arguments)]
pub fn from_csv(
    file_path: &str,
    delimiter: Option<char>,
    sep: Option<char>,
    comments: &str,
    data_structure: Option<&str>,
    directed: bool,
    bipartite: bool,
    weighted: bool,
    reindex: bool,
    shape: Option<(usize, usize)>,
    matrix_only: Option<bool>,
) -> Result<ParseResult, ParseError> {
    let (header_len, delim_guess, _comment_guess, ds_guess) =
        scan_header(file_path, delimiter.map(|c| c.to_string()).as_deref(), comments, 100)?;
    let delim = delimiter.or(sep).unwrap_or(delim_guess);
    let ds = data_structure
        .map(|s| s.to_string())
        .unwrap_or(ds_guess)
        .to_lowercase();

    if ds == "edge_list" {
        // Pass 1: determine mode (numeric vs labeled) and collect sizing/reindex metadata.
        let file = File::open(file_path).map_err(|_| ParseError::CsvIoError)?;
        let reader = BufReader::new(file);
        let mut try_numeric = true;
        let mut has_data = false;
        let mut max_u: i64 = -1;
        let mut max_v: i64 = -1;
        let mut node_ids = std::collections::HashSet::<i64>::new();
        let mut row_ids = std::collections::HashSet::<i64>::new();
        let mut col_ids = std::collections::HashSet::<i64>::new();
        for (i, line_res) in reader.lines().enumerate() {
            if i < header_len {
                continue;
            }
            let line = line_res.map_err(|_| ParseError::CsvIoError)?;
            if line.trim().is_empty() {
                continue;
            }
            has_data = true;
            let cols: Vec<&str> = line.split(delim).map(|s| s.trim()).collect();
            if cols.len() < 2 {
                return Err(ParseError::CsvFormatError);
            }
            if try_numeric {
                let parsed_u = cols[0].parse::<i64>();
                let parsed_v = cols[1].parse::<i64>();
                let parsed_w = if cols.len() >= 3 {
                    cols[2].parse::<f64>().is_ok()
                } else {
                    true
                };
                if parsed_u.is_ok() && parsed_v.is_ok() && parsed_w {
                    let u = parsed_u.map_err(|_| ParseError::CsvFormatError)?;
                    let v = parsed_v.map_err(|_| ParseError::CsvFormatError)?;
                    if bipartite {
                        row_ids.insert(u);
                        col_ids.insert(v);
                    } else {
                        node_ids.insert(u);
                        node_ids.insert(v);
                    }
                    if u > max_u {
                        max_u = u;
                    }
                    if v > max_v {
                        max_v = v;
                    }
                } else {
                    try_numeric = false;
                }
            }
        }
        if !has_data {
            return Err(ParseError::CsvFormatError);
        }

        if try_numeric {
            // Build optional reindex maps first to preserve stable sorted-id behavior.
            let mut map_nodes = std::collections::HashMap::<i64, usize>::new();
            let mut names: Option<Vec<i64>> = None;
            let mut map_rows = std::collections::HashMap::<i64, usize>::new();
            let mut map_cols = std::collections::HashMap::<i64, usize>::new();
            let mut names_row: Option<Vec<i64>> = None;
            let mut names_col: Option<Vec<i64>> = None;
            if reindex {
                if bipartite {
                    let mut sorted_rows: Vec<i64> = row_ids.into_iter().collect();
                    sorted_rows.sort_unstable();
                    for (idx, id) in sorted_rows.iter().copied().enumerate() {
                        map_rows.insert(id, idx);
                    }
                    let mut sorted_cols: Vec<i64> = col_ids.into_iter().collect();
                    sorted_cols.sort_unstable();
                    for (idx, id) in sorted_cols.iter().copied().enumerate() {
                        map_cols.insert(id, idx);
                    }
                    names_row = Some(sorted_rows);
                    names_col = Some(sorted_cols);
                } else {
                    let mut sorted_nodes: Vec<i64> = node_ids.into_iter().collect();
                    sorted_nodes.sort_unstable();
                    for (idx, id) in sorted_nodes.iter().copied().enumerate() {
                        map_nodes.insert(id, idx);
                    }
                    names = Some(sorted_nodes);
                }
            }

            // Determine matrix shape before streaming additions.
            if bipartite {
                let n_row = if reindex {
                    names_row.as_ref().map_or(0, Vec::len)
                } else {
                    (max_u.max(0) as usize).saturating_add(1)
                };
                let n_col = if reindex {
                    names_col.as_ref().map_or(0, Vec::len)
                } else {
                    (max_v.max(0) as usize).saturating_add(1)
                };
                let final_shape = shape.unwrap_or((n_row, n_col));
                let mut tri = TriMat::<f64>::new(final_shape);

                // Pass 2: stream and fill matrix.
                let file = File::open(file_path).map_err(|_| ParseError::CsvIoError)?;
                let reader = BufReader::new(file);
                for (i, line_res) in reader.lines().enumerate() {
                    if i < header_len {
                        continue;
                    }
                    let line = line_res.map_err(|_| ParseError::CsvIoError)?;
                    if line.trim().is_empty() {
                        continue;
                    }
                    let cols: Vec<&str> = line.split(delim).map(|s| s.trim()).collect();
                    if cols.len() < 2 {
                        return Err(ParseError::CsvFormatError);
                    }
                    let u = cols[0].parse::<i64>().map_err(|_| ParseError::CsvFormatError)?;
                    let v = cols[1].parse::<i64>().map_err(|_| ParseError::CsvFormatError)?;
                    if !reindex && (u < 0 || v < 0) {
                        return Err(ParseError::CsvFormatError);
                    }
                    let w = if cols.len() >= 3 {
                        cols[2].parse::<f64>().map_err(|_| ParseError::CsvFormatError)?
                    } else {
                        1.0
                    };
                    let ui = if reindex {
                        *map_rows.get(&u).ok_or(ParseError::CsvFormatError)?
                    } else {
                        u as usize
                    };
                    let vi = if reindex {
                        *map_cols.get(&v).ok_or(ParseError::CsvFormatError)?
                    } else {
                        v as usize
                    };
                    ensure_in_bounds(ui, vi, final_shape)?;
                    tri.add_triplet(ui, vi, if weighted { w } else { 1.0 });
                }
                let mat = tri.to_csr::<usize>();
                if matrix_only.unwrap_or(true) {
                    Ok(ParseResult::Matrix(mat))
                } else {
                    Ok(ParseResult::Dataset(GraphDataset {
                        adjacency: None,
                        biadjacency: Some(mat),
                        names: None,
                        names_str: None,
                        names_row,
                        names_col,
                        node_attribute: None,
                        edge_attribute: None,
                        meta: None,
                    }))
                }
            } else {
                let n = shape
                    .map(|s| s.0.max(s.1))
                    .unwrap_or_else(|| (max_u.max(max_v).max(0) as usize).saturating_add(1));
                let mut tri = TriMat::<f64>::new((n, n));

                // Pass 2: stream and fill matrix.
                let file = File::open(file_path).map_err(|_| ParseError::CsvIoError)?;
                let reader = BufReader::new(file);
                for (i, line_res) in reader.lines().enumerate() {
                    if i < header_len {
                        continue;
                    }
                    let line = line_res.map_err(|_| ParseError::CsvIoError)?;
                    if line.trim().is_empty() {
                        continue;
                    }
                    let cols: Vec<&str> = line.split(delim).map(|s| s.trim()).collect();
                    if cols.len() < 2 {
                        return Err(ParseError::CsvFormatError);
                    }
                    let u = cols[0].parse::<i64>().map_err(|_| ParseError::CsvFormatError)?;
                    let v = cols[1].parse::<i64>().map_err(|_| ParseError::CsvFormatError)?;
                    if !reindex && (u < 0 || v < 0) {
                        return Err(ParseError::CsvFormatError);
                    }
                    let w = if cols.len() >= 3 {
                        cols[2].parse::<f64>().map_err(|_| ParseError::CsvFormatError)?
                    } else {
                        1.0
                    };
                    let ui = if reindex {
                        *map_nodes.get(&u).ok_or(ParseError::CsvFormatError)?
                    } else {
                        u as usize
                    };
                    let vi = if reindex {
                        *map_nodes.get(&v).ok_or(ParseError::CsvFormatError)?
                    } else {
                        v as usize
                    };
                    ensure_in_bounds(ui, vi, (n, n))?;
                    tri.add_triplet(ui, vi, if weighted { w } else { 1.0 });
                }
                let mut mat = tri.to_csr::<usize>();
                if !directed {
                    mat = directed2undirected(&mat, weighted);
                }
                if matrix_only.unwrap_or(names.is_none()) {
                    Ok(ParseResult::Matrix(mat))
                } else {
                    Ok(ParseResult::Dataset(GraphDataset {
                        adjacency: Some(mat),
                        biadjacency: None,
                        names,
                        names_str: None,
                        names_row: None,
                        names_col: None,
                        node_attribute: None,
                        edge_attribute: None,
                        meta: None,
                    }))
                }
            }
        } else {
            if bipartite {
                return Err(ParseError::CsvFormatError);
            }

            // Labeled edge list: stream directly into tri using on-the-fly id assignment.
            let file = File::open(file_path).map_err(|_| ParseError::CsvIoError)?;
            let reader = BufReader::new(file);
            let mut names = Vec::<String>::new();
            let mut id_map = std::collections::HashMap::<String, usize>::new();
            let mut triplets = Vec::<(usize, usize, f64)>::new();
            for (i, line_res) in reader.lines().enumerate() {
                if i < header_len {
                    continue;
                }
                let line = line_res.map_err(|_| ParseError::CsvIoError)?;
                if line.trim().is_empty() {
                    continue;
                }
                let cols: Vec<&str> = line.split(delim).map(|s| s.trim()).collect();
                if cols.len() < 2 {
                    return Err(ParseError::CsvFormatError);
                }
                let s = cols[0].to_string();
                let t = cols[1].to_string();
                let w = if cols.len() >= 3 {
                    cols[2]
                        .parse::<f64>()
                        .map_err(|_| ParseError::CsvFormatError)?
                } else {
                    1.0
                };
                let u = if let Some(&id) = id_map.get(&s) {
                    id
                } else {
                    let id = names.len();
                    names.push(s.clone());
                    id_map.insert(s, id);
                    id
                };
                let v = if let Some(&id) = id_map.get(&t) {
                    id
                } else {
                    let id = names.len();
                    names.push(t.clone());
                    id_map.insert(t, id);
                    id
                };
                triplets.push((u, v, if weighted { w } else { 1.0 }));
            }
            let n = shape
                .map(|s| s.0.max(s.1))
                .unwrap_or(names.len());
            let mut tri = TriMat::<f64>::new((n, n));
            for (u, v, w) in triplets {
                ensure_in_bounds(u, v, (n, n))?;
                tri.add_triplet(u, v, w);
            }
            let mut mat = tri.to_csr::<usize>();
            if !directed {
                mat = directed2undirected(&mat, weighted);
            }
            if matrix_only.unwrap_or(false) {
                Ok(ParseResult::Matrix(mat))
            } else {
                Ok(ParseResult::Dataset(GraphDataset {
                    adjacency: Some(mat),
                    biadjacency: None,
                    names: None,
                    names_str: Some(names),
                    names_row: None,
                    names_col: None,
                    node_attribute: None,
                    edge_attribute: None,
                    meta: None,
                }))
            }
        }
    } else {
        // Adjacency list mode in two passes to avoid buffering all parsed rows.
        let file = File::open(file_path).map_err(|_| ParseError::CsvIoError)?;
        let reader = BufReader::new(file);
        let mut row_count = 0usize;
        let mut max_neighbor = 0usize;
        for (i, line_res) in reader.lines().enumerate() {
            if i < header_len {
                continue;
            }
            let line = line_res.map_err(|_| ParseError::CsvIoError)?;
            if line.trim().is_empty() {
                continue;
            }
            for x in line.split(delim).map(|s| s.trim()) {
                if x.is_empty() {
                    continue;
                }
                let neigh = x.parse::<i64>().map_err(|_| ParseError::CsvFormatError)?;
                if neigh < 0 {
                    return Err(ParseError::CsvFormatError);
                }
                let neigh_u = neigh as usize;
                if neigh_u > max_neighbor {
                    max_neighbor = neigh_u;
                }
            }
            row_count += 1;
        }
        if row_count == 0 {
            return Err(ParseError::CsvFormatError);
        }
        let default_n = row_count.max(max_neighbor.saturating_add(1));
        let n = shape.map(|s| s.0.max(s.1)).unwrap_or(default_n);
        let tri_shape = if bipartite {
            shape.unwrap_or((row_count, max_neighbor.saturating_add(1)))
        } else {
            (n, n)
        };
        let mut tri = TriMat::<f64>::new(tri_shape);

        let file = File::open(file_path).map_err(|_| ParseError::CsvIoError)?;
        let reader = BufReader::new(file);
        let mut src = 0usize;
        for (i, line_res) in reader.lines().enumerate() {
            if i < header_len {
                continue;
            }
            let line = line_res.map_err(|_| ParseError::CsvIoError)?;
            if line.trim().is_empty() {
                continue;
            }
            for x in line.split(delim).map(|s| s.trim()) {
                if x.is_empty() {
                    continue;
                }
                let dst = x.parse::<i64>().map_err(|_| ParseError::CsvFormatError)?;
                if dst < 0 {
                    return Err(ParseError::CsvFormatError);
                }
                ensure_in_bounds(src, dst as usize, tri_shape)?;
                tri.add_triplet(src, dst as usize, 1.0);
            }
            src += 1;
        }
        let mut mat = tri.to_csr::<usize>();
        if !directed && !bipartite {
            mat = directed2undirected(&mat, weighted);
        }
        if matrix_only.unwrap_or(true) {
            Ok(ParseResult::Matrix(mat))
        } else if bipartite {
            Ok(ParseResult::Dataset(GraphDataset {
                adjacency: None,
                biadjacency: Some(mat),
                names: None,
                names_str: None,
                names_row: None,
                names_col: None,
                node_attribute: None,
                edge_attribute: None,
                meta: None,
            }))
        } else {
            Ok(ParseResult::Dataset(GraphDataset {
                adjacency: Some(mat),
                biadjacency: None,
                names: None,
                names_str: None,
                names_row: None,
                names_col: None,
                node_attribute: None,
                edge_attribute: None,
                meta: None,
            }))
        }
    }
}

/// Loads one label string per line from a text file.
///
/// # Errors
/// Returns [`ParseError::CsvIoError`] on file read failure.
pub fn load_labels(file: &str) -> Result<Vec<String>, ParseError> {
    let f = File::open(file).map_err(|_| ParseError::CsvIoError)?;
    let reader = BufReader::new(f);
    let mut out = Vec::new();
    for line in reader.lines() {
        out.push(line.map_err(|_| ParseError::CsvIoError)?.trim().to_string());
    }
    Ok(out)
}

/// Reads graph flags from the first line of a Netset-style header file.
///
/// # Errors
/// Returns [`ParseError::CsvIoError`] on file read failure.
pub fn load_header(file: &str) -> Result<(bool, bool, bool), ParseError> {
    let f = File::open(file).map_err(|_| ParseError::CsvIoError)?;
    let mut reader = BufReader::new(f);
    let mut first = String::new();
    reader
        .read_line(&mut first)
        .map_err(|_| ParseError::CsvIoError)?;
    let directed = first.contains("asym");
    let bipartite = first.contains("bip");
    let weighted = !first.contains("unweighted");
    Ok((directed, bipartite, weighted))
}

/// Loads `key: value` metadata rows into a [`Dataset`].
///
/// # Arguments
/// - `file`: Path to the metadata file.
/// - `delimiter`: Separator between keys and values.
///
/// # Errors
/// Returns [`ParseError::CsvIoError`] on file read failure.
pub fn load_metadata(file: &str, delimiter: &str) -> Result<Dataset, ParseError> {
    let f = File::open(file).map_err(|_| ParseError::CsvIoError)?;
    let reader = BufReader::new(f);
    let mut metadata = Dataset::new();
    for line in reader.lines() {
        let row = line.map_err(|_| ParseError::CsvIoError)?;
        if let Some((k, v)) = row.split_once(delimiter) {
            metadata.set_attr(k.trim(), DatasetValue::Str(v.trim().to_string()));
        }
    }
    Ok(metadata)
}

/// Maps GraphML/Java type names to Rust scalar type labels.
pub fn java_type_to_rust_type(value: &str) -> Option<&'static str> {
    match value {
        "boolean" => Some("bool"),
        "int" => Some("int"),
        "string" => Some("string"),
        "long" | "float" | "double" => Some("float"),
        _ => None,
    }
}

/// Parses a GraphML file into a [`GraphDataset`].
///
/// # Arguments
/// - `file_path`: Path to the GraphML file.
/// - `weight_key`: Edge attribute name used for weights.
///
/// # Errors
/// Returns [`ParseError::GraphMlError`] on parse failure.
pub fn from_graphml(file_path: &str, weight_key: &str) -> Result<GraphDataset, ParseError> {
    let text = std::fs::read_to_string(file_path).map_err(|_| ParseError::GraphMlError)?;
    let graph_start = text.find("<graph").ok_or(ParseError::GraphMlError)?;
    let graph_end_tag = text[graph_start..]
        .find('>')
        .map(|i| graph_start + i)
        .ok_or(ParseError::GraphMlError)?;
    let graph_header = &text[graph_start..=graph_end_tag];
    let is_undirected = graph_header.contains("edgedefault=\"undirected\"");

    let mut weight_key_id: Option<String> = None;
    let mut node_attr_desc = Dataset::new();
    let mut edge_attr_desc = Dataset::new();
    let mut file_desc: Option<String> = None;
    if let Some(desc_start) = text.find("<desc>") {
        let start = desc_start + "<desc>".len();
        if let Some(end_rel) = text[start..].find("</desc>") {
            file_desc = Some(text[start..start + end_rel].trim().to_string());
        }
    }
    for chunk in text.match_indices("<key ") {
        let start = chunk.0;
        let tag_end = text[start..]
            .find('>')
            .map(|i| start + i)
            .ok_or(ParseError::GraphMlError)?;
        let key_tag = &text[start..=tag_end];
        let key_close = text[tag_end + 1..]
            .find("</key>")
            .map(|i| tag_end + 1 + i)
            .unwrap_or(tag_end);
        let key_body = if key_close > tag_end {
            &text[tag_end + 1..key_close]
        } else {
            ""
        };
        let attr_name = extract_attr(key_tag, "attr.name");
        if attr_name.as_deref() == Some(weight_key) {
            weight_key_id = extract_attr(key_tag, "id");
        }
        let key_for = extract_attr(key_tag, "for");
        if let Some(name) = attr_name {
            if let Some(desc_pos) = key_body.find("<desc>") {
                let ds = desc_pos + "<desc>".len();
                if let Some(de_rel) = key_body[ds..].find("</desc>") {
                    let desc = key_body[ds..ds + de_rel].trim().to_string();
                    match key_for.as_deref() {
                        Some("node") => node_attr_desc.set_attr(name, DatasetValue::Str(desc)),
                        Some("edge") => edge_attr_desc.set_attr(name, DatasetValue::Str(desc)),
                        _ => {}
                    }
                }
            }
        }
    }

    let mut node_names = Vec::<String>::new();
    let mut node_map = HashMap::<String, usize>::new();
    let canonical = graph_header.contains("parse.nodeids=\"canonical\"");
    for chunk in text.match_indices("<node ") {
        let start = chunk.0;
        let end = text[start..]
            .find('>')
            .map(|i| start + i)
            .ok_or(ParseError::GraphMlError)?;
        let node_tag = &text[start..=end];
        if let Some(id) = extract_attr(node_tag, "id") {
            let idx = node_names.len();
            if canonical && id.starts_with('n') {
                let parsed = id[1..].parse::<usize>().unwrap_or(idx);
                node_map.insert(id.clone(), parsed);
                if node_names.len() <= parsed {
                    node_names.resize(parsed + 1, String::new());
                }
                node_names[parsed] = id;
            } else {
                node_map.insert(id.clone(), idx);
                node_names.push(id);
            }
        }
    }
    let n = node_names.len();
    let mut tri = TriMat::<f64>::new((n, n));

    let mut offset = 0usize;
    while let Some(local_start) = text[offset..].find("<edge ") {
        let start = offset + local_start;
        let tag_end = text[start..]
            .find('>')
            .map(|i| start + i)
            .ok_or(ParseError::GraphMlError)?;
        let edge_tag = &text[start..=tag_end];
        let self_closing = edge_tag.trim_end().ends_with("/>");
        let (close, body) = if self_closing {
            (tag_end, "")
        } else {
            let close = text[tag_end + 1..]
                .find("</edge>")
                .map(|i| tag_end + 1 + i)
                .ok_or(ParseError::GraphMlError)?;
            (close, &text[tag_end + 1..close])
        };
        let src = extract_attr(edge_tag, "source").ok_or(ParseError::GraphMlError)?;
        let dst = extract_attr(edge_tag, "target").ok_or(ParseError::GraphMlError)?;
        let u = *node_map.get(&src).ok_or(ParseError::GraphMlError)?;
        let v = *node_map.get(&dst).ok_or(ParseError::GraphMlError)?;

        let mut w = 1.0;
        if let Some(weight_id) = &weight_key_id {
            for data_start in body.match_indices("<data ") {
                let ds = data_start.0;
                let de = body[ds..]
                    .find('>')
                    .map(|i| ds + i)
                    .ok_or(ParseError::GraphMlError)?;
                let data_tag = &body[ds..=de];
                if extract_attr(data_tag, "key").as_deref() == Some(weight_id.as_str()) {
                    let tail = &body[de + 1..];
                    if let Some(end_text) = tail.find("</data>") {
                        w = tail[..end_text].trim().parse::<f64>().unwrap_or(1.0);
                    }
                }
            }
        }
        tri.add_triplet(u, v, w);
        let directed_attr = extract_attr(edge_tag, "directed");
        let duplicate = directed_attr
            .map(|d| d != "true")
            .unwrap_or(is_undirected);
        if duplicate {
            tri.add_triplet(v, u, w);
        }
        offset = if self_closing {
            tag_end + 1
        } else {
            close + "</edge>".len()
        };
    }
    let adjacency = tri.to_csr::<usize>();
    Ok(GraphDataset {
        adjacency: Some(adjacency),
        biadjacency: None,
        names: None,
        names_str: Some(node_names),
        names_row: None,
        names_col: None,
        node_attribute: None,
        edge_attribute: None,
        meta: {
            let has_node = node_attr_desc != Dataset::new();
            let has_edge = edge_attr_desc != Dataset::new();
            if file_desc.is_none() && !has_node && !has_edge {
                None
            } else {
                let mut meta = Dataset::new();
                if let Some(desc) = file_desc {
                    meta.set_attr("description", DatasetValue::Str(desc));
                }
                if has_node {
                    meta.set_attr("node_attributes", DatasetValue::Str("present".to_string()));
                }
                if has_edge {
                    meta.set_attr("edge_attributes", DatasetValue::Str("present".to_string()));
                }
                Some(meta)
            }
        },
    })
}

fn extract_attr(tag: &str, attr: &str) -> Option<String> {
    for quote in ['"', '\''] {
        let needle = format!("{attr}={quote}");
        if let Some(start) = tag.find(&needle) {
            let rest = &tag[start + needle.len()..];
            if let Some(end) = rest.find(quote) {
                return Some(rest[..end].to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_list_and_adjacency_list() {
        let edge_list = vec![(0, 1, None), (1, 2, Some(2.0))];
        let out = from_edge_list(&edge_list, false, false, true, false, None, Some(true)).unwrap();
        match out {
            ParseResult::Matrix(a) => {
                assert_eq!(a.shape(), (3, 3));
                assert!(a.nnz() > 0);
            }
            _ => panic!("expected matrix"),
        }

        let adj_list = vec![vec![1, 2], vec![2, 3]];
        let out =
            from_adjacency_list(&adj_list, true, false, true, false, None, Some(true)).unwrap();
        match out {
            ParseResult::Matrix(a) => assert_eq!(a.shape(), (4, 4)),
            _ => panic!("expected matrix"),
        }
    }

    #[test]
    fn test_reindex_and_number() {
        let edge_list = vec![(14, 31, None), (42, 50, None), (0, 12, None)];
        let out = from_edge_list(&edge_list, false, false, true, true, None, Some(false)).unwrap();
        match out {
            ParseResult::Dataset(g) => {
                assert_eq!(g.names.unwrap_or_default(), vec![0, 12, 14, 31, 42, 50]);
            }
            _ => panic!("expected dataset"),
        }
        assert!(is_number("3"));
        assert!(!is_number("a"));
    }

    #[test]
    fn test_scan_and_csv_and_loaders() {
        let path = "stub_parse_data.txt";
        std::fs::write(path, "%stub\n1 3\n4 5\n0 2").expect("write stub");

        let (header, delim, comment, ds) = scan_header(path, None, "#%", 100).expect("scan");
        assert_eq!(header, 1);
        assert_eq!(delim, ' ');
        assert_eq!(comment, '%');
        assert_eq!(ds, "edge_list");

        let parsed = from_csv(
            path,
            None,
            None,
            "#%",
            None,
            false,
            false,
            true,
            false,
            None,
            Some(true),
        )
        .expect("parse csv");
        match parsed {
            ParseResult::Matrix(a) => assert_eq!(a.shape(), (6, 6)),
            _ => panic!("expected matrix"),
        }

        let labels_path = "stub_labels.txt";
        std::fs::write(labels_path, "a\nb\nc\n").expect("write labels");
        let labels = load_labels(labels_path).expect("labels");
        assert_eq!(labels, vec!["a".to_string(), "b".to_string(), "c".to_string()]);

        let head_path = "stub_head.txt";
        std::fs::write(head_path, "bip asym unweighted\n").expect("write header");
        let (directed, bipartite, weighted) = load_header(head_path).expect("header");
        assert!(directed);
        assert!(bipartite);
        assert!(!weighted);

        let meta_path = "stub_meta.txt";
        std::fs::write(meta_path, "name: demo\nversion: v1\n").expect("write metadata");
        let metadata = load_metadata(meta_path, ": ").expect("metadata");
        assert_eq!(
            metadata.get_attr("name"),
            Some(&DatasetValue::Str("demo".to_string()))
        );
        assert_eq!(
            metadata.get_attr("version"),
            Some(&DatasetValue::Str("v1".to_string()))
        );

        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(labels_path);
        let _ = std::fs::remove_file(head_path);
        let _ = std::fs::remove_file(meta_path);
    }

    #[test]
    fn test_csv_labeled_weighted_edge_list() {
        let path = "stub_labeled_weighted.txt";
        std::fs::write(path, "%stub\nf, e, 5\na, d, 6\nc, b, 1").expect("write csv");
        let parsed = from_csv(
            path,
            None,
            None,
            "#%",
            None,
            false,
            false,
            true,
            false,
            None,
            Some(false),
        )
        .expect("parse labeled edge list");
        match parsed {
            ParseResult::Dataset(g) => {
                let a = g.adjacency.expect("adjacency");
                let names = g.names_str.unwrap_or_default();
                assert_eq!(a.shape(), (6, 6));
                assert_eq!(names.len(), 6);
                assert!(a.nnz() > 0);
            }
            _ => panic!("expected dataset"),
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_graphml_basic() {
        let path = "stub_graphml.graphml";
        std::fs::write(
            path,
            r#"<?xml version='1.0' encoding='utf-8'?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="d0" for="edge" attr.name="weight" attr.type="int"/>
  <graph edgedefault="directed">
    <node id="node1"/>
    <node id="node2"/>
    <edge source="node1" target="node2">
      <data key="d0">1</data>
    </edge>
  </graph>
</graphml>"#,
        )
        .expect("write graphml");
        let g = from_graphml(path, "weight").expect("graphml parse");
        let a = g.adjacency.expect("adjacency");
        assert_eq!(a.shape(), (2, 2));
        assert_eq!(a.nnz(), 1);
        assert_eq!(g.names_str.unwrap_or_default(), vec!["node1", "node2"]);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_java_type_mapping() {
        assert_eq!(java_type_to_rust_type("boolean"), Some("bool"));
        assert_eq!(java_type_to_rust_type("int"), Some("int"));
        assert_eq!(java_type_to_rust_type("string"), Some("string"));
        assert_eq!(java_type_to_rust_type("double"), Some("float"));
        assert_eq!(java_type_to_rust_type("unknown"), None);
    }

    #[test]
    fn test_shape_bounds_error_no_panic() {
        let edge_list = vec![(0, 2, Some(1.0))];
        let out = from_edge_list(
            &edge_list,
            true,
            false,
            true,
            false,
            Some((2, 2)),
            Some(true),
        );
        assert!(matches!(out, Err(ParseError::CsvFormatError)));

        let path = "stub_parse_shape_bounds.txt";
        std::fs::write(path, "0 2\n").expect("write csv");
        let out = from_csv(
            path,
            Some(' '),
            None,
            "#%",
            Some("edge_list"),
            true,
            false,
            true,
            false,
            Some((2, 2)),
            Some(true),
        );
        assert!(matches!(out, Err(ParseError::CsvFormatError)));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_graphml_canonical_and_metadata() {
        let path = "stub_graphml_meta.graphml";
        std::fs::write(
            path,
            r#"<?xml version='1.0' encoding='utf-8'?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <desc>Some file</desc>
  <key id="d0" for="edge" attr.name="weight" attr.type="int"/>
  <key id="d1" for="node" attr.name="color" attr.type="string"><desc>Color</desc></key>
  <key id="d2" for="edge" attr.name="distance" attr.type="double"><desc>Distance</desc></key>
  <graph edgedefault="undirected" parse.nodeids="canonical">
    <node id="n0"/>
    <node id="n1"/>
    <node id="n2"/>
    <edge source="n0" target="n1"><data key="d0">1</data></edge>
    <edge source="n1" target="n2" directed="true"><data key="d0">1</data></edge>
  </graph>
</graphml>"#,
        )
        .expect("write graphml");
        let g = from_graphml(path, "weight").expect("graphml parse");
        let a = g.adjacency.expect("adjacency");
        assert_eq!(a.shape(), (3, 3));
        assert_eq!(g.names_str.unwrap_or_default(), vec!["n0", "n1", "n2"]);
        let meta = g.meta.expect("meta");
        assert_eq!(
            meta.get_attr("description"),
            Some(&DatasetValue::Str("Some file".to_string()))
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_graphml_single_quotes_and_self_closing_edges() {
        let path = "stub_graphml_single_quote.graphml";
        std::fs::write(
            path,
            r#"<?xml version='1.0' encoding='utf-8'?>
<graphml xmlns='http://graphml.graphdrawing.org/xmlns'>
  <key id='d0' for='edge' attr.name='weight' attr.type='double'/>
  <graph edgedefault='directed'>
    <node id='n0'/>
    <node id='n1'/>
    <node id='n2'/>
    <edge source='n0' target='n1'/>
    <edge source='n1' target='n2'><data key='d0'>2.5</data></edge>
  </graph>
</graphml>"#,
        )
        .expect("write graphml");
        let g = from_graphml(path, "weight").expect("graphml parse");
        let a = g.adjacency.expect("adjacency");
        assert_eq!(a.shape(), (3, 3));
        assert_eq!(a.nnz(), 2);
        assert_eq!(a.get(0, 1), Some(&1.0));
        assert_eq!(a.get(1, 2), Some(&2.5));
        let _ = std::fs::remove_file(path);
    }
}
