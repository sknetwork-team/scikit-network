use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Component, Path, PathBuf};
use std::process::Command;

use npyz::sparse::Csr as NpyzCsr;
use npyz::NpyFile;
use sprs::{CsMat, TriMat};

use crate::data::parse::GraphDataset;
use crate::data::parse::{from_csv, from_graphml, load_header, load_labels, load_metadata, ParseResult};

/// Errors raised by dataset loading and saving.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadError {
    /// File or directory I/O failed.
    Io,
    /// Parsed file contents are invalid.
    Format,
    /// Archive extraction path escapes the target directory.
    PathTraversal,
    /// Requested loader is not implemented.
    NotImplemented,
    /// Dataset folder contains no loadable matrix.
    InvalidDataset,
    /// Remote dataset download failed.
    Network,
    /// Archive listing or extraction failed.
    Archive,
    /// Required external tool (`curl`, `tar`) is missing or failed.
    ExternalTool,
}

/// Resolves and creates the scikit-network data home directory.
///
/// # Arguments
/// - `data_home`: Explicit directory override (`None` uses env vars or default).
///
/// # Errors
/// Returns [`LoadError::Io`] when the directory cannot be created.
pub fn get_data_home(data_home: Option<&Path>) -> Result<PathBuf, LoadError> {
    let path = if let Some(p) = data_home {
        p.to_path_buf()
    } else if let Ok(p) = env::var("SCIKIT_NETWORK_DATA") {
        PathBuf::from(p)
    } else if let Ok(home) = env::var("HOME") {
        PathBuf::from(home).join("scikit_network_data")
    } else {
        PathBuf::from("scikit_network_data")
    };
    if !path.exists() {
        fs::create_dir_all(&path).map_err(|_| LoadError::Io)?;
    }
    Ok(path)
}

/// Removes the data home directory and all cached datasets.
///
/// # Errors
/// Returns [`LoadError::Io`] on removal failure.
pub fn clear_data_home(data_home: Option<&Path>) -> Result<(), LoadError> {
    let path = get_data_home(data_home)?;
    if path.exists() {
        fs::remove_dir_all(path).map_err(|_| LoadError::Io)?;
    }
    Ok(())
}

/// Deletes files inside the data home directory but keeps the folder.
///
/// # Errors
/// Returns [`LoadError::Io`] on directory read or file removal failure.
pub fn clean_data_home(data_home: Option<&Path>) -> Result<(), LoadError> {
    let path = get_data_home(data_home)?;
    for entry in fs::read_dir(path).map_err(|_| LoadError::Io)? {
        let entry = entry.map_err(|_| LoadError::Io)?;
        let p = entry.path();
        if p.is_file() {
            fs::remove_file(p).map_err(|_| LoadError::Io)?;
        }
    }
    Ok(())
}

/// Write a dataset to *folder* as CSR text files (``adjacency.tsv``, optional ``names*.txt``).
///
/// This is the fast native reload format for Rust. It is **not** the NumPy ``.npz`` layout used by
/// Python ``scikit-network`` ``save`` / ``load_netset``.
///
/// # Errors
/// Returns [`LoadError::Io`] on file creation or write failure.
pub fn save(folder: &Path, data: &GraphDataset) -> Result<(), LoadError> {
    if folder.exists() {
        fs::remove_dir_all(folder).map_err(|_| LoadError::Io)?;
    }
    fs::create_dir_all(folder).map_err(|_| LoadError::Io)?;
    if let Some(a) = &data.adjacency {
        write_csr(folder.join("adjacency.tsv").as_path(), a)?;
    }
    if let Some(b) = &data.biadjacency {
        write_csr(folder.join("biadjacency.tsv").as_path(), b)?;
    }
    if let Some(names) = &data.names {
        write_i64_vec(folder.join("names.txt").as_path(), names)?;
    }
    if let Some(names) = &data.names_str {
        write_str_vec(folder.join("names_str.txt").as_path(), names)?;
    }
    if let Some(names_row) = &data.names_row {
        write_i64_vec(folder.join("names_row.txt").as_path(), names_row)?;
    }
    if let Some(names_col) = &data.names_col {
        write_i64_vec(folder.join("names_col.txt").as_path(), names_col)?;
    }
    Ok(())
}

/// Load a CSR text bundle from *folder* (see [`save`]).
///
/// # Errors
/// Returns [`LoadError::Io`] or [`LoadError::Format`] on read or parse failure.
pub fn load(folder: &Path) -> Result<GraphDataset, LoadError> {
    if !folder.exists() {
        return Err(LoadError::Io);
    }
    let adjacency = read_csr_if_exists(folder.join("adjacency.tsv").as_path())?;
    let biadjacency = read_csr_if_exists(folder.join("biadjacency.tsv").as_path())?;
    let names = read_i64_vec_if_exists(folder.join("names.txt").as_path())?;
    let names_str = read_str_vec_if_exists(folder.join("names_str.txt").as_path())?;
    let names_row = read_i64_vec_if_exists(folder.join("names_row.txt").as_path())?;
    let names_col = read_i64_vec_if_exists(folder.join("names_col.txt").as_path())?;
    Ok(GraphDataset {
        adjacency,
        biadjacency,
        names,
        names_str,
        names_row,
        names_col,
        node_attribute: None,
        edge_attribute: None,
        meta: None,
    })
}

/// Save a CSR text bundle under ``get_data_home(data_home) / bundle_name`` (see [`save`]).
///
/// # Errors
/// Returns the same errors as [`save`].
pub fn save_csr_bundle(
    data: &GraphDataset,
    bundle_name: &str,
    data_home: Option<&Path>,
) -> Result<(), LoadError> {
    let root = get_data_home(data_home)?;
    let folder = root.join(bundle_name);
    save(&folder, data)
}

/// Load a CSR text bundle from ``get_data_home(data_home) / bundle_name`` (see [`load`]).
///
/// # Errors
/// Returns the same errors as [`load`].
pub fn load_csr_bundle(
    bundle_name: &str,
    data_home: Option<&Path>,
) -> Result<GraphDataset, LoadError> {
    let root = get_data_home(data_home)?;
    let folder = root.join(bundle_name);
    load(&folder)
}

/// Options controlling NetSet / folder loads (see [`load_netset_with_options`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LoadOptions {
    /// Load only the adjacency/biadjacency matrix; skip labels, names, and metadata sidecars.
    pub adjacency_only: bool,
    /// After loading matrices from NPZ/GraphML/CSV, write ``adjacency.tsv`` / ``biadjacency.tsv``
    /// beside the source files for faster subsequent reloads.
    pub materialize_csr: bool,
}

/// Returns whether `target` resolves inside `directory` (path-traversal guard).
pub fn is_within_directory(directory: &Path, target: &Path) -> bool {
    let base = normalize_lexical(directory);
    let tgt = normalize_lexical(target);
    tgt.starts_with(&base)
}

/// Validates archive member paths stay within `base_dir`.
///
/// # Errors
/// Returns [`LoadError::PathTraversal`] when a member escapes the base directory.
pub fn validate_extract_paths(base_dir: &Path, members: &[PathBuf]) -> Result<(), LoadError> {
    for m in members {
        let full = base_dir.join(m);
        if !is_within_directory(base_dir, &full) {
            return Err(LoadError::PathTraversal);
        }
    }
    Ok(())
}

/// Downloads and loads a Netset dataset by name with default options.
///
/// # Arguments
/// - `name`: Netset dataset name (`None` returns [`LoadError::InvalidDataset`]).
/// - `data_home`: Optional data-home override.
///
/// # Errors
/// Returns [`LoadError`] variants for missing names, I/O, network, or format failures.
pub fn load_netset(name: Option<&str>, data_home: Option<&Path>) -> Result<GraphDataset, LoadError> {
    load_netset_with_options(name, data_home, &LoadOptions::default())
}

/// Downloads and loads a Netset dataset with explicit [`LoadOptions`].
///
/// # Errors
/// Returns the same errors as [`load_netset`].
pub fn load_netset_with_options(
    name: Option<&str>,
    data_home: Option<&Path>,
    opts: &LoadOptions,
) -> Result<GraphDataset, LoadError> {
    const NETSET_URL: &str = "https://netset.telecom-paris.fr";
    let Some(raw_name) = name else {
        return Err(LoadError::InvalidDataset);
    };
    let name = raw_name.to_lowercase();
    let root = get_data_home(data_home)?;
    let netset_dir = root.join("netset");
    let direct_path = root.join(&name);
    let dataset_path = if direct_path.exists() {
        direct_path
    } else {
        netset_dir.join(&name)
    };
    if !dataset_path.exists() {
        fs::create_dir_all(&netset_dir).map_err(|_| LoadError::Io)?;
        let archive_name = format!("{name}_npz.tar.gz");
        let archive_path = netset_dir.join(&archive_name);
        let archive_url = format!("{NETSET_URL}/datasets_npz/{archive_name}");
        download_file(&archive_url, &archive_path)?;
        fs::create_dir_all(&dataset_path).map_err(|_| LoadError::Io)?;
        let members = list_archive_members(&archive_path)?;
        validate_extract_paths(&dataset_path, &members)?;
        extract_archive(&archive_path, &dataset_path)?;
        let _ = fs::remove_file(&archive_path);
    }

    load_dataset_folder(&dataset_path, opts)
}

/// Load a dataset folder (NetSet NPZ layout, CSR text bundle, or hybrid).
///
/// # Arguments
/// - `dataset_path`: Folder containing matrices and optional sidecars.
/// - `opts`: Controls sidecar loading and CSR materialization.
///
/// # Errors
/// Returns [`LoadError::InvalidDataset`] when no matrix is found.
pub fn load_dataset_folder(dataset_path: &Path, opts: &LoadOptions) -> Result<GraphDataset, LoadError> {
    let mut dataset = empty_graph_dataset();
    let mut matrices_from_npz = false;

    dataset.adjacency = read_csr_if_exists(&dataset_path.join("adjacency.tsv"))?;
    dataset.biadjacency = read_csr_if_exists(&dataset_path.join("biadjacency.tsv"))?;

    if dataset.adjacency.is_none() && dataset.biadjacency.is_none() {
        matrices_from_npz = ingest_matrix_files(dataset_path, &mut dataset)?;
    }

    if !opts.adjacency_only {
        ingest_sidecar_files(dataset_path, &mut dataset)?;
        merge_text_sidecars(dataset_path, &mut dataset)?;
    }

    if dataset.adjacency.is_none() && dataset.biadjacency.is_none() {
        return Err(LoadError::InvalidDataset);
    }

    if opts.materialize_csr && matrices_from_npz {
        materialize_csr_files(dataset_path, &dataset)?;
    }

    Ok(dataset)
}

fn is_label_text_file(name: &str) -> bool {
    matches!(
        name,
        "names.txt" | "names_str.txt" | "names_row.txt" | "names_col.txt"
    )
}

fn empty_graph_dataset() -> GraphDataset {
    GraphDataset {
        adjacency: None,
        biadjacency: None,
        names: None,
        names_str: None,
        names_row: None,
        names_col: None,
        node_attribute: None,
        edge_attribute: None,
        meta: None,
    }
}

fn ingest_matrix_files(dataset_path: &Path, dataset: &mut GraphDataset) -> Result<bool, LoadError> {
    let mut from_npz = false;
    for entry in fs::read_dir(dataset_path).map_err(|_| LoadError::Io)? {
        let entry = entry.map_err(|_| LoadError::Io)?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let file_name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_lowercase();
        if file_name == "adjacency.tsv" || file_name == "biadjacency.tsv" {
            continue;
        }
        if file_name.ends_with(".graphml") {
            let parsed = from_graphml(path.to_str().ok_or(LoadError::Io)?, "weight")
                .map_err(|_| LoadError::Format)?;
            merge_dataset(dataset, parsed);
        } else if file_name.ends_with(".npz") {
            let mat = load_npz_matrix(path.to_str().ok_or(LoadError::Io)?)?;
            from_npz = true;
            assign_matrix(dataset, &file_name, mat);
        } else if (file_name.ends_with(".csv")
            || (file_name.ends_with(".tsv")
                && !file_name.starts_with("adjacency")
                && !file_name.starts_with("biadjacency"))
            || file_name.ends_with(".txt"))
            && !is_label_text_file(&file_name)
        {
            let (directed, bipartite, weighted) =
                load_header(path.to_str().ok_or(LoadError::Io)?).unwrap_or((false, false, true));
            let parsed = from_csv(
                path.to_str().ok_or(LoadError::Io)?,
                None,
                None,
                "#%",
                None,
                directed,
                bipartite,
                weighted,
                false,
                None,
                Some(false),
            )
            .map_err(|_| LoadError::Format)?;
            match parsed {
                ParseResult::Matrix(m) => {
                    if bipartite {
                        dataset.biadjacency = Some(m);
                    } else {
                        dataset.adjacency = Some(m);
                    }
                }
                ParseResult::Dataset(g) => merge_dataset(dataset, g),
            }
        }
    }
    Ok(from_npz)
}

fn assign_matrix(dataset: &mut GraphDataset, file_name: &str, mat: CsMat<f64>) {
    if file_name.contains("biadjacency") {
        dataset.biadjacency = Some(mat);
    } else if file_name.contains("adjacency") {
        dataset.adjacency = Some(mat);
    } else if mat.rows() == mat.cols() && dataset.adjacency.is_none() {
        dataset.adjacency = Some(mat);
    } else if dataset.biadjacency.is_none() {
        dataset.biadjacency = Some(mat);
    }
}

fn ingest_sidecar_files(dataset_path: &Path, dataset: &mut GraphDataset) -> Result<(), LoadError> {
    for entry in fs::read_dir(dataset_path).map_err(|_| LoadError::Io)? {
        let entry = entry.map_err(|_| LoadError::Io)?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let file_name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_lowercase();
        if file_name.ends_with(".npy") {
            let labels = load_npy_labels(path.to_str().ok_or(LoadError::Io)?)?;
            assign_npy_labels(dataset, &file_name, labels)?;
        } else if file_name.ends_with(".p") {
            continue;
        } else if file_name.starts_with("metadata") {
            let meta = load_metadata(path.to_str().ok_or(LoadError::Io)?, ": ")
                .map_err(|_| LoadError::Format)?;
            dataset.meta = Some(meta);
        } else if file_name.starts_with("names")
            && (file_name.ends_with(".csv")
                || file_name.ends_with(".tsv")
                || file_name.ends_with(".txt"))
        {
            let labels = load_labels(path.to_str().ok_or(LoadError::Io)?).map_err(|_| LoadError::Format)?;
            if file_name.contains("names_row") {
                dataset.names_row = Some(parse_labels_i64(&labels)?);
            } else if file_name.contains("names_col") {
                dataset.names_col = Some(parse_labels_i64(&labels)?);
            } else if let Ok(nums) = try_parse_labels_i64(&labels) {
                dataset.names = Some(nums);
            } else {
                dataset.names_str = Some(labels);
            }
        }
    }
    Ok(())
}

fn assign_npy_labels(
    dataset: &mut GraphDataset,
    file_name: &str,
    labels: ParsedLabels,
) -> Result<(), LoadError> {
    if file_name.contains("names_row") {
        match labels {
            ParsedLabels::Int(v) => dataset.names_row = Some(v),
            ParsedLabels::Str(_) => return Err(LoadError::Format),
        }
    } else if file_name.contains("names_col") {
        match labels {
            ParsedLabels::Int(v) => dataset.names_col = Some(v),
            ParsedLabels::Str(_) => return Err(LoadError::Format),
        }
    } else {
        match labels {
            ParsedLabels::Int(v) => dataset.names = Some(v),
            ParsedLabels::Str(v) => dataset.names_str = Some(v),
        }
    }
    Ok(())
}

fn merge_text_sidecars(dataset_path: &Path, dataset: &mut GraphDataset) -> Result<(), LoadError> {
    if dataset.names.is_none() {
        dataset.names = read_i64_vec_if_exists(&dataset_path.join("names.txt"))?;
    }
    if dataset.names_str.is_none() {
        dataset.names_str = read_str_vec_if_exists(&dataset_path.join("names_str.txt"))?;
    }
    if dataset.names_row.is_none() {
        dataset.names_row = read_i64_vec_if_exists(&dataset_path.join("names_row.txt"))?;
    }
    if dataset.names_col.is_none() {
        dataset.names_col = read_i64_vec_if_exists(&dataset_path.join("names_col.txt"))?;
    }
    Ok(())
}

fn materialize_csr_files(folder: &Path, dataset: &GraphDataset) -> Result<(), LoadError> {
    if let Some(a) = &dataset.adjacency {
        let path = folder.join("adjacency.tsv");
        if !path.exists() {
            write_csr(&path, a)?;
        }
    }
    if let Some(b) = &dataset.biadjacency {
        let path = folder.join("biadjacency.tsv");
        if !path.exists() {
            write_csr(&path, b)?;
        }
    }
    Ok(())
}

fn write_csr(path: &Path, mat: &CsMat<f64>) -> Result<(), LoadError> {
    let mut file = File::create(path).map_err(|_| LoadError::Io)?;
    writeln!(file, "{} {}", mat.rows(), mat.cols()).map_err(|_| LoadError::Io)?;
    for (i, row) in mat.outer_iterator().enumerate() {
        for (&j, &v) in row.indices().iter().zip(row.data().iter()) {
            writeln!(file, "{i}\t{j}\t{v}").map_err(|_| LoadError::Io)?;
        }
    }
    Ok(())
}

fn read_csr_if_exists(path: &Path) -> Result<Option<CsMat<f64>>, LoadError> {
    if !path.exists() {
        return Ok(None);
    }
    let file = File::open(path).map_err(|_| LoadError::Io)?;
    let mut reader = BufReader::new(file);
    let mut header = String::new();
    reader.read_line(&mut header).map_err(|_| LoadError::Io)?;
    let dims: Vec<_> = header.split_whitespace().collect();
    if dims.len() != 2 {
        return Err(LoadError::Format);
    }
    let n = dims[0].parse::<usize>().map_err(|_| LoadError::Format)?;
    let m = dims[1].parse::<usize>().map_err(|_| LoadError::Format)?;
    let mut tri = TriMat::<f64>::new((n, m));
    for line in reader.lines() {
        let line = line.map_err(|_| LoadError::Io)?;
        if line.trim().is_empty() {
            continue;
        }
        let cols: Vec<_> = line.split('\t').collect();
        if cols.len() != 3 {
            return Err(LoadError::Format);
        }
        let i = cols[0].parse::<usize>().map_err(|_| LoadError::Format)?;
        let j = cols[1].parse::<usize>().map_err(|_| LoadError::Format)?;
        let v = cols[2].parse::<f64>().map_err(|_| LoadError::Format)?;
        tri.add_triplet(i, j, v);
    }
    Ok(Some(tri.to_csr::<usize>()))
}

fn write_i64_vec(path: &Path, values: &[i64]) -> Result<(), LoadError> {
    let mut file = File::create(path).map_err(|_| LoadError::Io)?;
    for v in values {
        writeln!(file, "{v}").map_err(|_| LoadError::Io)?;
    }
    Ok(())
}

fn write_str_vec(path: &Path, values: &[String]) -> Result<(), LoadError> {
    let mut file = File::create(path).map_err(|_| LoadError::Io)?;
    for v in values {
        writeln!(file, "{v}").map_err(|_| LoadError::Io)?;
    }
    Ok(())
}

fn read_i64_vec_if_exists(path: &Path) -> Result<Option<Vec<i64>>, LoadError> {
    if !path.exists() {
        return Ok(None);
    }
    let file = File::open(path).map_err(|_| LoadError::Io)?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for line in reader.lines() {
        let line = line.map_err(|_| LoadError::Io)?;
        out.push(line.parse::<i64>().map_err(|_| LoadError::Format)?);
    }
    Ok(Some(out))
}

fn read_str_vec_if_exists(path: &Path) -> Result<Option<Vec<String>>, LoadError> {
    if !path.exists() {
        return Ok(None);
    }
    let file = File::open(path).map_err(|_| LoadError::Io)?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for line in reader.lines() {
        out.push(line.map_err(|_| LoadError::Io)?);
    }
    Ok(Some(out))
}

fn normalize_lexical(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            _ => out.push(c.as_os_str()),
        }
    }
    out
}

fn merge_dataset(target: &mut GraphDataset, parsed: GraphDataset) {
    if target.adjacency.is_none() {
        target.adjacency = parsed.adjacency;
    }
    if target.biadjacency.is_none() {
        target.biadjacency = parsed.biadjacency;
    }
    if target.names.is_none() {
        target.names = parsed.names;
    }
    if target.names_str.is_none() {
        target.names_str = parsed.names_str;
    }
    if target.names_row.is_none() {
        target.names_row = parsed.names_row;
    }
    if target.names_col.is_none() {
        target.names_col = parsed.names_col;
    }
    if target.node_attribute.is_none() {
        target.node_attribute = parsed.node_attribute;
    }
    if target.edge_attribute.is_none() {
        target.edge_attribute = parsed.edge_attribute;
    }
    if target.meta.is_none() {
        target.meta = parsed.meta;
    }
}

fn download_file(url: &str, output_path: &Path) -> Result<(), LoadError> {
    let status = Command::new("curl")
        .arg("-fsSL")
        .arg(url)
        .arg("-o")
        .arg(output_path)
        .status()
        .map_err(|_| LoadError::ExternalTool)?;
    if status.success() {
        Ok(())
    } else {
        Err(LoadError::Network)
    }
}

fn list_archive_members(archive_path: &Path) -> Result<Vec<PathBuf>, LoadError> {
    let output = Command::new("tar")
        .arg("-tzf")
        .arg(archive_path)
        .output()
        .map_err(|_| LoadError::ExternalTool)?;
    if !output.status.success() {
        return Err(LoadError::Archive);
    }
    let text = String::from_utf8(output.stdout).map_err(|_| LoadError::Archive)?;
    let members = text
        .lines()
        .filter(|s| !s.trim().is_empty())
        .map(PathBuf::from)
        .collect();
    Ok(members)
}

fn extract_archive(archive_path: &Path, destination: &Path) -> Result<(), LoadError> {
    let status = Command::new("tar")
        .arg("-xzf")
        .arg(archive_path)
        .arg("-C")
        .arg(destination)
        .status()
        .map_err(|_| LoadError::ExternalTool)?;
    if status.success() {
        Ok(())
    } else {
        Err(LoadError::Archive)
    }
}

fn csr_to_sprs(csr: NpyzCsr<f64>) -> Result<CsMat<f64>, LoadError> {
    let nrows = csr.shape[0] as usize;
    let ncols = csr.shape[1] as usize;
    let indptr: Vec<usize> = csr.indptr.iter().map(|&x| x as usize).collect();
    let indices: Vec<usize> = csr.indices.iter().map(|&x| x as usize).collect();
    Ok(CsMat::new((nrows, ncols), indptr, indices, csr.data))
}

fn load_npz_matrix(path: &str) -> Result<CsMat<f64>, LoadError> {
    let mut npz = npyz::npz::NpzArchive::open(path).map_err(|_| LoadError::Format)?;
    if let Ok(csr) = NpyzCsr::<f64>::from_npz(&mut npz) {
        return csr_to_sprs(csr);
    }
    let mut npz = npyz::npz::NpzArchive::open(path).map_err(|_| LoadError::Format)?;
    if let Ok(csr) = NpyzCsr::<bool>::from_npz(&mut npz) {
        let nrows = csr.shape[0] as usize;
        let ncols = csr.shape[1] as usize;
        let indptr: Vec<usize> = csr.indptr.iter().map(|&x| x as usize).collect();
        let indices: Vec<usize> = csr.indices.iter().map(|&x| x as usize).collect();
        let data: Vec<f64> = csr.data.into_iter().map(|b| if b { 1.0 } else { 0.0 }).collect();
        return Ok(CsMat::new((nrows, ncols), indptr, indices, data));
    }
    Err(LoadError::Format)
}

enum ParsedLabels {
    Int(Vec<i64>),
    Str(Vec<String>),
}

fn load_npy_labels(path: &str) -> Result<ParsedLabels, LoadError> {
    let bytes = fs::read(path).map_err(|_| LoadError::Io)?;
    load_npy_labels_from_bytes(&bytes)
}

fn load_npy_labels_from_bytes(bytes: &[u8]) -> Result<ParsedLabels, LoadError> {
    let reopen = || NpyFile::new(bytes).map_err(|_| LoadError::Format);
    if let Ok(vals) = reopen()?.into_vec::<i64>() {
        return Ok(ParsedLabels::Int(vals));
    }
    if let Ok(vals) = reopen()?.into_vec::<i32>() {
        return Ok(ParsedLabels::Int(vals.into_iter().map(i64::from).collect()));
    }
    if let Ok(strs) = reopen()?.into_vec::<String>() {
        return Ok(ParsedLabels::Str(strs));
    }
    let floats = reopen()?.into_vec::<f64>().map_err(|_| LoadError::Format)?;
    Ok(ParsedLabels::Int(
        floats.into_iter().map(|x| x as i64).collect(),
    ))
}

fn parse_labels_i64(labels: &[String]) -> Result<Vec<i64>, LoadError> {
    try_parse_labels_i64(labels).map_err(|_| LoadError::Format)
}

fn try_parse_labels_i64(labels: &[String]) -> Result<Vec<i64>, LoadError> {
    let mut out = Vec::with_capacity(labels.len());
    for s in labels {
        out.push(s.parse::<i64>().map_err(|_| LoadError::Format)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_home_and_clean() {
        let root = PathBuf::from("stub_data_home");
        let _ = clear_data_home(Some(&root));
        let home = get_data_home(Some(&root)).expect("home");
        assert!(home.exists());
        std::fs::write(home.join("junk.txt"), "x").expect("write");
        clean_data_home(Some(&root)).expect("clean");
        assert!(!home.join("junk.txt").exists());
        clear_data_home(Some(&root)).expect("clear");
        assert!(!home.exists());
    }

    #[test]
    fn test_save_load_roundtrip() {
        let folder = PathBuf::from("stub_bundle");
        let mut tri = TriMat::<f64>::new((3, 3));
        tri.add_triplet(0, 1, 1.0);
        tri.add_triplet(1, 2, 2.0);
        let data = GraphDataset {
            adjacency: Some(tri.to_csr::<usize>()),
            biadjacency: None,
            names: Some(vec![0, 1, 2]),
            names_str: Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
            names_row: None,
            names_col: None,
            node_attribute: None,
            edge_attribute: None,
            meta: None,
        };
        save(&folder, &data).expect("save");
        let loaded = load(&folder).expect("load");
        assert_eq!(loaded.names, data.names);
        assert_eq!(loaded.names_str, data.names_str);
        assert_eq!(
            loaded.adjacency.expect("a").data().to_vec(),
            data.adjacency.expect("b").data().to_vec()
        );
        let _ = std::fs::remove_dir_all(folder);
    }

    #[test]
    fn test_csr_bundle_at_data_home() {
        let home = PathBuf::from("stub_bundle_home");
        let _ = clear_data_home(Some(&home));
        let data = GraphDataset {
            adjacency: None,
            biadjacency: None,
            names: Some(vec![1, 2]),
            names_str: Some(vec!["u".to_string(), "v".to_string()]),
            names_row: None,
            names_col: None,
            node_attribute: None,
            edge_attribute: None,
            meta: None,
        };
        save_csr_bundle(&data, "bundle1", Some(&home)).expect("save bundle");
        let loaded = load_csr_bundle("bundle1", Some(&home)).expect("load bundle");
        assert_eq!(loaded.names, data.names);
        assert_eq!(loaded.names_str, data.names_str);
        let _ = clear_data_home(Some(&home));
    }

    #[test]
    fn test_within_directory_and_extract_paths() {
        let base = PathBuf::from("a/b");
        assert!(is_within_directory(&base, &PathBuf::from("a/b/c/file.txt")));
        assert!(!is_within_directory(&base, &PathBuf::from("a/b/../../etc/passwd")));

        let ok_members = vec![PathBuf::from("x.txt"), PathBuf::from("dir/y.txt")];
        assert!(validate_extract_paths(&base, &ok_members).is_ok());
        let bad_members = vec![PathBuf::from("../escape.txt")];
        assert!(matches!(
            validate_extract_paths(&base, &bad_members),
            Err(LoadError::PathTraversal)
        ));
    }

    #[test]
    fn test_load_netset_missing_remote_or_local_data_errors() {
        assert!(load_netset(Some("wikivitals"), None).is_err());
    }

    #[test]
    fn test_load_netset_local_cached_bundle() {
        let home = PathBuf::from("stub_netset_home");
        let _ = clear_data_home(Some(&home));
        let dataset_path = get_data_home(Some(&home))
            .expect("home")
            .join("netset")
            .join("stub");
        std::fs::create_dir_all(&dataset_path).expect("mkdir");
        let mut tri = TriMat::<f64>::new((2, 2));
        tri.add_triplet(0, 1, 1.0);
        let data = GraphDataset {
            adjacency: Some(tri.to_csr::<usize>()),
            biadjacency: None,
            names: None,
            names_str: Some(vec!["a".to_string(), "b".to_string()]),
            names_row: None,
            names_col: None,
            node_attribute: None,
            edge_attribute: None,
            meta: None,
        };
        save(&dataset_path, &data).expect("save");

        let loaded = load_netset(Some("stub"), Some(&home)).expect("load netset");
        assert!(loaded.adjacency.is_some());
        assert_eq!(loaded.names_str, data.names_str);
        let _ = clear_data_home(Some(&home));
    }

    #[test]
    fn test_adjacency_only_skips_sidecars() {
        let folder = PathBuf::from("stub_adj_only");
        let _ = std::fs::remove_dir_all(&folder);
        let mut tri = TriMat::<f64>::new((2, 2));
        tri.add_triplet(0, 1, 1.0);
        let data = GraphDataset {
            adjacency: Some(tri.to_csr::<usize>()),
            biadjacency: None,
            names: Some(vec![10, 20]),
            names_str: Some(vec!["a".to_string(), "b".to_string()]),
            names_row: None,
            names_col: None,
            node_attribute: None,
            edge_attribute: None,
            meta: None,
        };
        save(&folder, &data).expect("save");
        let opts = LoadOptions {
            adjacency_only: true,
            materialize_csr: false,
        };
        let loaded = load_dataset_folder(&folder, &opts).expect("load adjacency only");
        assert!(loaded.adjacency.is_some());
        assert!(loaded.names.is_none());
        assert!(loaded.names_str.is_none());
        let _ = std::fs::remove_dir_all(folder);
    }

    #[test]
    fn test_materialize_csr_writes_tsv_beside_npz() {
        let src = PathBuf::from("benchmarking/datasets/bundles/karate/adjacency.npz");
        if !src.is_file() {
            return;
        }
        let folder = PathBuf::from("stub_materialize_npz");
        let _ = std::fs::remove_dir_all(&folder);
        std::fs::create_dir_all(&folder).expect("mkdir");
        std::fs::copy(&src, folder.join("adjacency.npz")).expect("copy npz");
        let opts = LoadOptions {
            adjacency_only: true,
            materialize_csr: true,
        };
        load_dataset_folder(&folder, &opts).expect("load and materialize");
        assert!(folder.join("adjacency.tsv").is_file());
        let reloaded = load_dataset_folder(&folder, &opts).expect("reload from tsv");
        assert!(reloaded.adjacency.is_some());
        let _ = std::fs::remove_dir_all(folder);
    }

    #[test]
    fn test_hybrid_tsv_matrix_with_text_sidecars() {
        let folder = PathBuf::from("stub_hybrid_sidecars");
        let _ = std::fs::remove_dir_all(&folder);
        let mut tri = TriMat::<f64>::new((2, 2));
        tri.add_triplet(0, 1, 1.0);
        save(
            &folder,
            &GraphDataset {
                adjacency: Some(tri.to_csr::<usize>()),
                biadjacency: None,
                names: Some(vec![1, 2]),
                names_str: None,
                names_row: None,
                names_col: None,
                node_attribute: None,
                edge_attribute: None,
                meta: None,
            },
        )
        .expect("save");
        let loaded = load_dataset_folder(&folder, &LoadOptions::default()).expect("hybrid load");
        assert_eq!(loaded.names, Some(vec![1, 2]));
        assert!(loaded.adjacency.is_some());
        let _ = std::fs::remove_dir_all(folder);
    }

    #[test]
    fn test_load_netset_skips_pickle_when_no_matrix_present() {
        let home = PathBuf::from("stub_netset_pickle");
        let _ = clear_data_home(Some(&home));
        let dataset_path = get_data_home(Some(&home))
            .expect("home")
            .join("netset")
            .join("stubpickle");
        std::fs::create_dir_all(&dataset_path).expect("mkdir");
        std::fs::write(dataset_path.join("names.p"), b"pickle").expect("write");
        assert!(matches!(
            load_netset(Some("stubpickle"), Some(&home)),
            Err(LoadError::InvalidDataset)
        ));
        let _ = clear_data_home(Some(&home));
    }
}
