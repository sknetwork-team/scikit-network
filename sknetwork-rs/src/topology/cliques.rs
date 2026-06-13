use sprs::CsMat;

use crate::path::dag::{DagError, get_dag};
use crate::topology::core::{CoreError, get_core_decomposition};

#[derive(Debug, Clone, PartialEq, Eq)]
/// Errors raised by clique error operations.
pub enum CliqueError {
    /// Indicates invalid clique size.
    InvalidCliqueSize,
    /// Indicates invalid core order.
    InvalidCoreOrder,
    /// Indicates core.
    Core(CoreError),
    /// Indicates dag.
    Dag(DagError),
}

struct ListingState {
    ns: Vec<usize>,
    subs: Vec<Vec<usize>>,
    degrees: Vec<Vec<usize>>,
    lab: Vec<usize>,
}

impl ListingState {
    fn new(out_degrees: &[usize], clique_size: usize) -> Self {
        let n = out_degrees.len();
        let mut ns = vec![0; clique_size + 1];
        ns[clique_size] = n;

        let mut subs = vec![vec![0; n]; clique_size + 1];
        for (i, slot) in subs[clique_size].iter_mut().enumerate() {
            *slot = i;
        }

        let mut degrees = vec![vec![0; n]; clique_size + 1];
        degrees[clique_size].copy_from_slice(out_degrees);

        Self {
            ns,
            subs,
            degrees,
            lab: vec![clique_size; n],
        }
    }
}

fn count_cliques_from_level(
    neighbors: &[Vec<usize>],
    state: &mut ListingState,
    clique_size: usize,
) -> usize {
    if clique_size == 2 {
        let mut n_cliques = 0usize;
        for i in 0..state.ns[2] {
            let u = state.subs[2][i];
            n_cliques += state.degrees[2][u];
        }
        return n_cliques;
    }

    let mut n_cliques = 0usize;
    for i in 0..state.ns[clique_size] {
        let u = state.subs[clique_size][i];
        state.ns[clique_size - 1] = 0;

        for &v in neighbors[u].iter().take(state.degrees[clique_size][u]) {
            if state.lab[v] == clique_size {
                state.lab[v] = clique_size - 1;
                let idx = state.ns[clique_size - 1];
                state.subs[clique_size - 1][idx] = v;
                state.ns[clique_size - 1] += 1;
                state.degrees[clique_size - 1][v] = 0;
            }
        }

        for j in 0..state.ns[clique_size - 1] {
            let v = state.subs[clique_size - 1][j];
            let mut deg = 0usize;
            for &w in neighbors[v].iter().take(state.degrees[clique_size][v]) {
                if state.lab[w] == clique_size - 1 {
                    deg += 1;
                }
            }
            state.degrees[clique_size - 1][v] = deg;
        }

        n_cliques += count_cliques_from_level(neighbors, state, clique_size - 1);

        for j in 0..state.ns[clique_size - 1] {
            let v = state.subs[clique_size - 1][j];
            state.lab[v] = clique_size;
        }
    }

    n_cliques
}

fn clique_dag_order(
    adjacency: &CsMat<f64>,
    core_order: Option<&[i32]>,
) -> Result<CsMat<f64>, CliqueError> {
    let order: Vec<i32> = if let Some(order) = core_order {
        if order.len() != adjacency.rows() {
            return Err(CliqueError::InvalidCoreOrder);
        }
        order.to_vec()
    } else {
        let core = get_core_decomposition(adjacency).map_err(CliqueError::Core)?;
        // Match Python ``get_dag(adjacency, order=np.argsort(core))``: ``order[i]`` is the
        // i-th entry of ``argsort(core)`` (sorted node-index list), not per-node rank.
        let mut argsort: Vec<usize> = (0..core.len()).collect();
        // NumPy ``argsort(core)`` defaults to quicksort (unstable); stable Rust sort breaks parity.
        argsort.sort_unstable_by_key(|&i| core[i]);
        argsort.iter().map(|&node| node as i32).collect()
    };
    get_dag(adjacency, None, Some(&order)).map_err(CliqueError::Dag)
}

/// Counts cliques with core order.
pub fn count_cliques_with_core_order(
    adjacency: &CsMat<f64>,
    clique_size: usize,
    core_order: Option<&[i32]>,
) -> Result<usize, CliqueError> {
    if clique_size < 2 {
        return Err(CliqueError::InvalidCliqueSize);
    }
    if adjacency.rows() == 0 {
        return Ok(0);
    }
    let dag = clique_dag_order(adjacency, core_order)?;

    if clique_size == 2 {
        return Ok(dag.nnz());
    }

    let mut neighbors = Vec::with_capacity(dag.rows());
    let mut out_degrees = vec![0usize; dag.rows()];
    for i in 0..dag.rows() {
        if let Some(row) = dag.outer_view(i) {
            let inds = row.indices().to_vec();
            out_degrees[i] = inds.len();
            neighbors.push(inds);
        } else {
            neighbors.push(Vec::new());
        }
    }

    let mut state = ListingState::new(&out_degrees, clique_size);
    Ok(count_cliques_from_level(&neighbors, &mut state, clique_size))
}

/// Counts cliques.
///
/// # Errors
///
/// Returns [`CliqueError`] on failure.
pub fn count_cliques(adjacency: &CsMat<f64>, clique_size: usize) -> Result<usize, CliqueError> {
    count_cliques_with_core_order(adjacency, clique_size, None)
}

#[cfg(test)]
mod tests {
    use sprs::TriMat;

    use super::*;

    fn comb(n: usize, k: usize) -> usize {
        if k > n {
            return 0;
        }
        let mut num = 1usize;
        let mut den = 1usize;
        for i in 0..k {
            num *= n - i;
            den *= i + 1;
        }
        num / den
    }

    fn clique_graph(n: usize) -> CsMat<f64> {
        let mut tri = TriMat::<f64>::new((n, n));
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    tri.add_triplet(i, j, 1.0);
                }
            }
        }
        tri.to_csr::<usize>()
    }

    #[test]
    fn test_empty() {
        let empty = CsMat::<f64>::zero((4, 4));
        assert_eq!(count_cliques(&empty, 3).unwrap(), 0);
        assert_eq!(
            count_cliques(&empty, 1),
            Err(CliqueError::InvalidCliqueSize)
        );
    }

    #[test]
    fn test_disconnected() {
        let mut tri = TriMat::<f64>::new((5, 5));
        tri.add_triplet(0, 1, 1.0);
        tri.add_triplet(1, 0, 1.0);
        tri.add_triplet(1, 2, 1.0);
        tri.add_triplet(2, 1, 1.0);
        tri.add_triplet(0, 2, 1.0);
        tri.add_triplet(2, 0, 1.0);
        tri.add_triplet(3, 4, 1.0);
        tri.add_triplet(4, 3, 1.0);
        let adjacency = tri.to_csr::<usize>();
        assert_eq!(count_cliques(&adjacency, 3).unwrap(), 1);
    }

    #[test]
    fn test_cliques() {
        let n = 7;
        let adjacency = clique_graph(n);
        assert_eq!(count_cliques(&adjacency, 3).unwrap(), comb(n, 3));
        assert_eq!(count_cliques(&adjacency, 4).unwrap(), comb(n, 4));
    }
}
