use std::collections::HashMap;

/// Type alias for dendrogram.
pub type Dendrogram = Vec<[f64; 4]>;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Errors raised by hierarchy error operations.
pub enum HierarchyError {
    /// Indicates invalid dendrogram.
    InvalidDendrogram,
    /// Indicates invalid nclusters.
    InvalidNClusters,
    /// Indicates invalid max cluster size.
    InvalidMaxClusterSize,
}

fn check_dendrogram(dendrogram: &Dendrogram) -> Result<(), HierarchyError> {
    if dendrogram.is_empty() {
        return Err(HierarchyError::InvalidDendrogram);
    }
    let n = dendrogram.len() + 1;
    let mut prev_height = f64::NEG_INFINITY;
    for (t, row) in dendrogram.iter().enumerate() {
        let i = row[0];
        let j = row[1];
        let h = row[2];
        let size = row[3];
        if !i.is_finite()
            || !j.is_finite()
            || i < 0.0
            || j < 0.0
            || !size.is_finite()
            || size < 2.0
            || size > n as f64
        {
            return Err(HierarchyError::InvalidDendrogram);
        }
        let ii = i as usize;
        let jj = j as usize;
        if (ii as f64 - i).abs() > 0.0 || (jj as f64 - j).abs() > 0.0 || ii == jj {
            return Err(HierarchyError::InvalidDendrogram);
        }
        let max_node = n + t;
        if ii >= max_node || jj >= max_node {
            return Err(HierarchyError::InvalidDendrogram);
        }
        if h.is_nan() {
            return Err(HierarchyError::InvalidDendrogram);
        }
        if h < prev_height {
            return Err(HierarchyError::InvalidDendrogram);
        }
        prev_height = h;
    }
    if (dendrogram.last().map(|r| r[3]).unwrap_or(0.0) - n as f64).abs() > 0.0 {
        return Err(HierarchyError::InvalidDendrogram);
    }
    Ok(())
}

/// Computes split dendrogram.
pub fn split_dendrogram(dendrogram: &Dendrogram, shape: (usize, usize)) -> (Dendrogram, Dendrogram) {
    let (n1, n2) = shape;
    let mut dendrogram_row: Dendrogram = Vec::new();
    let mut dendrogram_col: Dendrogram = Vec::new();

    let mut id_row_new = n1 as i64;
    let mut id_col_new = n2 as i64;

    let mut size_row: HashMap<i64, i64> = (0..n1 as i64).map(|i| (i, 1)).collect();
    let mut size_col: HashMap<i64, i64> = (n1 as i64..(n1 + n2) as i64).map(|i| (i, 1)).collect();
    let mut id_row: HashMap<i64, i64> = (0..n1 as i64).map(|i| (i, i)).collect();
    let mut id_col: HashMap<i64, i64> = (n1 as i64..(n1 + n2) as i64)
        .enumerate()
        .map(|(k, i)| (i, k as i64))
        .collect();

    for (t, row) in dendrogram.iter().enumerate() {
        let i = row[0] as i64;
        let j = row[1] as i64;
        let height = row[2];
        let new_key = (n1 + n2 + t) as i64;

        if id_row.contains_key(&i) && id_row.contains_key(&j) {
            let si = size_row.remove(&i).unwrap_or(1);
            let sj = size_row.remove(&j).unwrap_or(1);
            size_row.insert(new_key, si + sj);
            let ii = id_row.remove(&i).unwrap_or(i);
            let jj = id_row.remove(&j).unwrap_or(j);
            dendrogram_row.push([ii as f64, jj as f64, height, (si + sj) as f64]);
            id_row.insert(new_key, id_row_new);
            id_row_new += 1;
        } else if id_row.contains_key(&i) {
            let si = size_row.remove(&i).unwrap_or(1);
            size_row.insert(new_key, si);
            let ii = id_row.remove(&i).unwrap_or(i);
            id_row.insert(new_key, ii);
        } else if id_row.contains_key(&j) {
            let sj = size_row.remove(&j).unwrap_or(1);
            size_row.insert(new_key, sj);
            let jj = id_row.remove(&j).unwrap_or(j);
            id_row.insert(new_key, jj);
        }

        if id_col.contains_key(&i) && id_col.contains_key(&j) {
            let si = size_col.remove(&i).unwrap_or(1);
            let sj = size_col.remove(&j).unwrap_or(1);
            size_col.insert(new_key, si + sj);
            let ii = id_col.remove(&i).unwrap_or(i);
            let jj = id_col.remove(&j).unwrap_or(j);
            dendrogram_col.push([ii as f64, jj as f64, height, (si + sj) as f64]);
            id_col.insert(new_key, id_col_new);
            id_col_new += 1;
        } else if id_col.contains_key(&i) {
            let si = size_col.remove(&i).unwrap_or(1);
            size_col.insert(new_key, si);
            let ii = id_col.remove(&i).unwrap_or(i);
            id_col.insert(new_key, ii);
        } else if id_col.contains_key(&j) {
            let sj = size_col.remove(&j).unwrap_or(1);
            size_col.insert(new_key, sj);
            let jj = id_col.remove(&j).unwrap_or(j);
            id_col.insert(new_key, jj);
        }
    }

    (dendrogram_row, dendrogram_col)
}

/// Computes reorder dendrogram.
pub fn reorder_dendrogram(dendrogram: &Dendrogram) -> Dendrogram {
    if dendrogram.is_empty() {
        return Vec::new();
    }
    let n = dendrogram.len() + 1;
    let mut with_keys: Vec<(usize, f64, f64)> = dendrogram
        .iter()
        .enumerate()
        .map(|(idx, row)| (idx, row[0].max(row[1]), row[2]))
        .collect();
    with_keys.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
    });
    let mut dendrogram_new: Dendrogram = with_keys.iter().map(|(idx, _, _)| dendrogram[*idx]).collect();
    let mut index_new: Vec<usize> = (0..(2 * n - 1)).collect();
    for (new_pos, (old_pos, _, _)) in with_keys.iter().enumerate() {
        index_new[n + *old_pos] = n + new_pos;
    }
    for row in &mut dendrogram_new {
        row[0] = index_new[row[0] as usize] as f64;
        row[1] = index_new[row[1] as usize] as f64;
    }
    dendrogram_new
}

#[derive(Debug, Clone)]
/// Tree enum.
pub enum Tree {
    /// Indicates leaf.
    Leaf(usize),
    /// Indicates node.
    Node(Vec<Tree>),
}

impl Tree {
    fn max_index(&self) -> usize {
        match self {
            Tree::Leaf(i) => *i,
            Tree::Node(children) => children.iter().map(|t| t.max_index()).max().unwrap_or(0),
        }
    }
}

/// Returns dendrogram.
pub fn get_dendrogram(tree: &Tree) -> Dendrogram {
    get_dendrogram_with_reorder(tree, true)
}

/// Returns dendrogram with reorder.
pub fn get_dendrogram_with_reorder(tree: &Tree, reorder: bool) -> Dendrogram {
    let mut dendrogram = Vec::<[f64; 4]>::new();
    let mut index = tree.max_index();
    let mut size_map = HashMap::<usize, usize>::new();
    let _ = get_dendrogram_rec(tree, &mut dendrogram, &mut index, 0, &mut size_map);
    if reorder {
        reorder_dendrogram(&dendrogram)
    } else {
        dendrogram
    }
}

fn get_dendrogram_rec(
    tree: &Tree,
    dendrogram: &mut Dendrogram,
    index: &mut usize,
    depth: usize,
    size_map: &mut HashMap<usize, usize>,
) -> usize {
    match tree {
        Tree::Leaf(i) => {
            size_map.insert(*i, 1);
            *i
        }
        Tree::Node(children) => {
            if children.len() == 1 {
                return get_dendrogram_rec(&children[0], dendrogram, index, depth, size_map);
            }
            let mut child_ids: Vec<usize> = children
                .iter()
                .map(|c| get_dendrogram_rec(c, dendrogram, index, depth + 1, size_map))
                .collect();
            let first = child_ids.remove(0);
            let second = child_ids.remove(0);
            let mut current_size = size_map.get(&first).copied().unwrap_or(1)
                + size_map.get(&second).copied().unwrap_or(1);
            *index += 1;
            let mut current = *index;
            dendrogram.push([
                first as f64,
                second as f64,
                -(depth as f64),
                current_size as f64,
            ]);
            size_map.insert(current, current_size);
            for cid in child_ids {
                current_size += size_map.get(&cid).copied().unwrap_or(1);
                *index += 1;
                dendrogram.push([
                    current as f64,
                    cid as f64,
                    -(depth as f64),
                    current_size as f64,
                ]);
                current = *index;
                size_map.insert(current, current_size);
            }
            current
        }
    }
}

/// Computes cut straight.
pub fn cut_straight(
    dendrogram: &Dendrogram,
    n_clusters: Option<usize>,
    threshold: Option<f64>,
    sort_clusters: bool,
) -> Result<Vec<usize>, HierarchyError> {
    check_dendrogram(dendrogram)?;
    let n = dendrogram.len() + 1;
    let target = if let Some(k) = n_clusters { k } else if threshold.is_some() { n } else { 2 };
    if target == 0 || target > n {
        return Err(HierarchyError::InvalidNClusters);
    }

    let mut heights: Vec<f64> = dendrogram.iter().map(|r| r[2]).collect();
    heights.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut cut = heights[n - target];
    if let Some(th) = threshold {
        if th > cut {
            cut = th;
        }
    }

    let mut cluster: HashMap<usize, Vec<usize>> = (0..n).map(|i| (i, vec![i])).collect();
    for (t, row) in dendrogram.iter().enumerate() {
        let i = row[0] as usize;
        let j = row[1] as usize;
        if row[2] < cut && cluster.contains_key(&i) && cluster.contains_key(&j) {
            let mut ci = cluster.remove(&i).unwrap_or_default();
            let mut cj = cluster.remove(&j).unwrap_or_default();
            ci.append(&mut cj);
            cluster.insert(n + t, ci);
        }
    }
    Ok(clusters_to_labels(cluster, n, sort_clusters))
}

/// Computes cut balanced.
pub fn cut_balanced(
    dendrogram: &Dendrogram,
    max_cluster_size: usize,
    sort_clusters: bool,
) -> Result<Vec<usize>, HierarchyError> {
    check_dendrogram(dendrogram)?;
    let n = dendrogram.len() + 1;
    if max_cluster_size < 2 || max_cluster_size > n {
        return Err(HierarchyError::InvalidMaxClusterSize);
    }
    let mut cluster: HashMap<usize, Vec<usize>> = (0..n).map(|i| (i, vec![i])).collect();
    for (t, row) in dendrogram.iter().enumerate() {
        let i = row[0] as usize;
        let j = row[1] as usize;
        if cluster.contains_key(&i) && cluster.contains_key(&j) {
            let size = cluster.get(&i).map(|x| x.len()).unwrap_or(0)
                + cluster.get(&j).map(|x| x.len()).unwrap_or(0);
            if size <= max_cluster_size {
                let mut ci = cluster.remove(&i).unwrap_or_default();
                let mut cj = cluster.remove(&j).unwrap_or_default();
                ci.append(&mut cj);
                cluster.insert(n + t, ci);
            }
        }
    }
    Ok(clusters_to_labels(cluster, n, sort_clusters))
}

/// Computes aggregate dendrogram.
pub fn aggregate_dendrogram(
    dendrogram: &Dendrogram,
    n_clusters: usize,
    return_counts: bool,
) -> Result<(Dendrogram, Option<Vec<usize>>), HierarchyError> {
    check_dendrogram(dendrogram)?;
    let n_nodes = dendrogram.len() + 1;
    if n_clusters == 0 || n_clusters > n_nodes {
        return Err(HierarchyError::InvalidNClusters);
    }
    let start = n_nodes - n_clusters;
    let mut new_dendrogram = dendrogram[start..].to_vec();
    let mut node_indices: Vec<usize> = new_dendrogram
        .iter()
        .flat_map(|r| [r[0] as usize, r[1] as usize])
        .collect();
    node_indices.sort_unstable();
    node_indices.dedup();
    let new_index: HashMap<usize, usize> = node_indices
        .iter()
        .enumerate()
        .map(|(i, &ix)| (ix, i))
        .collect();

    for row in &mut new_dendrogram {
        row[0] = *new_index.get(&(row[0] as usize)).unwrap_or(&0) as f64;
        row[1] = *new_index.get(&(row[1] as usize)).unwrap_or(&0) as f64;
    }

    if return_counts {
        let leaves: Vec<usize> = node_indices.into_iter().take(n_clusters).collect();
        let mut counts = Vec::with_capacity(leaves.len());
        for leaf in leaves {
            if leaf < n_nodes {
                counts.push(1);
            } else {
                counts.push(dendrogram[leaf - n_nodes][3] as usize);
            }
        }
        Ok((new_dendrogram, Some(counts)))
    } else {
        Ok((new_dendrogram, None))
    }
}

fn clusters_to_labels(
    cluster: HashMap<usize, Vec<usize>>,
    n: usize,
    sort_clusters: bool,
) -> Vec<usize> {
    let mut clusters: Vec<Vec<usize>> = cluster.into_values().collect();
    if sort_clusters {
        clusters.sort_by_key(|c| std::cmp::Reverse(c.len()));
    }
    let mut labels = vec![0usize; n];
    for (label, nodes) in clusters.iter().enumerate() {
        for &node in nodes {
            labels[node] = label;
        }
    }
    labels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_dendrogram_basic() {
        let d = vec![
            [0.0, 1.0, 0.0, 2.0],
            [2.0, 3.0, 0.0, 2.0],
            [4.0, 5.0, 1.0, 4.0],
        ];
        let (dr, dc) = split_dendrogram(&d, (2, 2));
        assert_eq!(dr.len(), 1);
        assert_eq!(dc.len(), 1);
        assert_eq!(dr[0][2], 0.0);
        assert_eq!(dc[0][2], 0.0);
    }

    #[test]
    fn test_check_dendrogram_rejects_malformed() {
        // repeated children
        let d = vec![[0.0, 0.0, 0.1, 2.0], [2.0, 1.0, 0.2, 3.0]];
        assert!(matches!(
            cut_straight(&d, Some(2), None, true),
            Err(HierarchyError::InvalidDendrogram)
        ));

        // invalid node index progression
        let d = vec![[0.0, 3.0, 0.1, 2.0], [2.0, 1.0, 0.2, 3.0]];
        assert!(matches!(
            cut_straight(&d, Some(2), None, true),
            Err(HierarchyError::InvalidDendrogram)
        ));

        // non-monotonic heights
        let d = vec![[0.0, 1.0, 1.0, 2.0], [2.0, 3.0, 0.5, 4.0]];
        assert!(matches!(
            cut_straight(&d, Some(2), None, true),
            Err(HierarchyError::InvalidDendrogram)
        ));
    }

    #[test]
    fn test_cut_straight() {
        let d = vec![
            [0.0, 1.0, 0.0, 2.0],
            [2.0, 3.0, 1.0, 2.0],
            [4.0, 5.0, 2.0, 4.0],
        ];
        let labels = cut_straight(&d, None, None, true).expect("cut");
        assert_eq!(labels.len(), 4);
        let n_clusters = labels.iter().copied().max().unwrap_or(0) + 1;
        assert_eq!(n_clusters, 2);
    }

    #[test]
    fn test_cut_balanced() {
        let d = vec![
            [0.0, 1.0, 0.0, 2.0],
            [2.0, 3.0, 0.2, 2.0],
            [4.0, 5.0, 1.0, 4.0],
        ];
        let labels = cut_balanced(&d, 2, true).expect("balanced");
        assert_eq!(labels.len(), 4);
        let n_clusters = labels.iter().copied().max().unwrap_or(0) + 1;
        assert_eq!(n_clusters, 2);
    }

    #[test]
    fn test_aggregate_dendrogram() {
        let d = vec![
            [0.0, 1.0, 0.0, 2.0],
            [2.0, 3.0, 1.0, 2.0],
            [4.0, 5.0, 2.0, 4.0],
        ];
        let (agg, counts) = aggregate_dendrogram(&d, 3, true).expect("agg");
        assert_eq!(agg.len(), 2);
        assert_eq!(counts.unwrap_or_default().len(), 3);
    }

    #[test]
    fn test_reorder_and_get_dendrogram() {
        let tree = Tree::Node(vec![
            Tree::Node(vec![Tree::Leaf(0), Tree::Leaf(1)]),
            Tree::Leaf(2),
        ]);
        let d = get_dendrogram(&tree);
        assert_eq!(d.len(), 2);
        assert!(d[0][2] <= d[1][2]);
    }
}
