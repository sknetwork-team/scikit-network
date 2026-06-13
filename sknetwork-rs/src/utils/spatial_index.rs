use std::cmp::Ordering;
use std::collections::BinaryHeap;

fn dist2(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

#[derive(Debug, Clone)]
struct Node {
    idx: usize,
    axis: usize,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
}

#[derive(Debug, Clone)]
/// KDTree value.
pub struct KDTree {
    points: Vec<Vec<f64>>,
    dim: usize,
    root: Option<Box<Node>>,
}

#[derive(Debug, Clone)]
/// KDTreeRef value.
pub struct KDTreeRef<'a> {
    points: &'a [Vec<f64>],
    dim: usize,
    root: Option<Box<Node>>,
}

#[derive(Debug, Clone, PartialEq)]
struct HeapEntry {
    dist2: f64,
    idx: usize,
}

impl Eq for HeapEntry {}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist2
            .total_cmp(&other.dist2)
            .then_with(|| self.idx.cmp(&other.idx))
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl KDTree {
    /// Computes build.
    pub fn build(points: Vec<Vec<f64>>) -> Option<Self> {
        if points.is_empty() || points[0].is_empty() {
            return None;
        }
        let dim = points[0].len();
        if points.iter().any(|p| p.len() != dim) {
            return None;
        }
        let mut indices: Vec<usize> = (0..points.len()).collect();
        let root = Self::build_rec(&points, &mut indices, 0, dim);
        Some(Self { points, dim, root })
    }

    fn build_rec(points: &[Vec<f64>], indices: &mut [usize], depth: usize, dim: usize) -> Option<Box<Node>> {
        if indices.is_empty() {
            return None;
        }
        let axis = depth % dim;
        indices.sort_by(|&a, &b| points[a][axis].total_cmp(&points[b][axis]));
        let mid = indices.len() / 2;
        let (left, right_with_mid) = indices.split_at_mut(mid);
        let (mid_slice, right) = right_with_mid.split_at_mut(1);
        let idx = mid_slice[0];
        Some(Box::new(Node {
            idx,
            axis,
            left: Self::build_rec(points, left, depth + 1, dim),
            right: Self::build_rec(points, right, depth + 1, dim),
        }))
    }

    /// Computes radius query.
    pub fn radius_query(&self, query: &[f64], radius: f64) -> Vec<usize> {
        let mut out = Vec::new();
        self.radius_query_into(query, radius, &mut out);
        out
    }

    /// Computes radius query into.
    pub fn radius_query_into(&self, query: &[f64], radius: f64, out: &mut Vec<usize>) {
        out.clear();
        if radius <= 0.0 || query.len() != self.dim {
            return;
        }
        let r2 = radius * radius;
        Self::radius_rec(&self.points, &self.root, query, r2, out);
    }

    fn radius_rec(
        points: &[Vec<f64>],
        node: &Option<Box<Node>>,
        query: &[f64],
        r2: f64,
        out: &mut Vec<usize>,
    ) {
        let Some(n) = node else {
            return;
        };
        let p = &points[n.idx];
        if dist2(p, query) <= r2 {
            out.push(n.idx);
        }
        let axis = n.axis;
        let delta = query[axis] - p[axis];
        let (near, far) = if delta <= 0.0 {
            (&n.left, &n.right)
        } else {
            (&n.right, &n.left)
        };
        Self::radius_rec(points, near, query, r2, out);
        if delta * delta <= r2 {
            Self::radius_rec(points, far, query, r2, out);
        }
    }

    /// Computes knn query.
    pub fn knn_query(&self, query: &[f64], k: usize) -> Vec<usize> {
        if k == 0 || query.len() != self.dim {
            return Vec::new();
        }
        let mut heap = BinaryHeap::<HeapEntry>::new();
        Self::knn_rec(&self.points, &self.root, query, k, &mut heap);
        let mut out: Vec<HeapEntry> = heap.into_vec();
        out.sort_by(|a, b| a.dist2.total_cmp(&b.dist2).then_with(|| a.idx.cmp(&b.idx)));
        out.into_iter().map(|e| e.idx).collect()
    }

    fn knn_rec(
        points: &[Vec<f64>],
        node: &Option<Box<Node>>,
        query: &[f64],
        k: usize,
        heap: &mut BinaryHeap<HeapEntry>,
    ) {
        let Some(n) = node else {
            return;
        };
        let p = &points[n.idx];
        let d2 = dist2(p, query);
        if heap.len() < k {
            heap.push(HeapEntry { dist2: d2, idx: n.idx });
        } else if let Some(top) = heap.peek()
            && d2 < top.dist2
        {
            let _ = heap.pop();
            heap.push(HeapEntry { dist2: d2, idx: n.idx });
        }
        let axis = n.axis;
        let delta = query[axis] - p[axis];
        let (near, far) = if delta <= 0.0 {
            (&n.left, &n.right)
        } else {
            (&n.right, &n.left)
        };
        Self::knn_rec(points, near, query, k, heap);
        let boundary = heap.peek().map(|e| e.dist2).unwrap_or(f64::INFINITY);
        if heap.len() < k || delta * delta <= boundary {
            Self::knn_rec(points, far, query, k, heap);
        }
    }
}

impl<'a> KDTreeRef<'a> {
    /// Computes build.
    pub fn build(points: &'a [Vec<f64>]) -> Option<Self> {
        if points.is_empty() || points[0].is_empty() {
            return None;
        }
        let dim = points[0].len();
        if points.iter().any(|p| p.len() != dim) {
            return None;
        }
        let mut indices: Vec<usize> = (0..points.len()).collect();
        let root = KDTree::build_rec(points, &mut indices, 0, dim);
        Some(Self { points, dim, root })
    }

    /// Computes radius query into.
    pub fn radius_query_into(&self, query: &[f64], radius: f64, out: &mut Vec<usize>) {
        out.clear();
        if radius <= 0.0 || query.len() != self.dim {
            return;
        }
        let r2 = radius * radius;
        KDTree::radius_rec(self.points, &self.root, query, r2, out);
    }
}

/// Computes brute radius.
pub fn brute_radius(points: &[Vec<f64>], query: &[f64], radius: f64) -> Vec<usize> {
    let mut out = Vec::new();
    brute_radius_into(points, query, radius, &mut out);
    out
}

/// Computes brute radius into.
pub fn brute_radius_into(points: &[Vec<f64>], query: &[f64], radius: f64, out: &mut Vec<usize>) {
    out.clear();
    if radius <= 0.0 {
        return;
    }
    let r2 = radius * radius;
    for (i, p) in points.iter().enumerate() {
        if dist2(p, query) <= r2 {
            out.push(i);
        }
    }
}

/// Computes brute knn.
pub fn brute_knn(points: &[Vec<f64>], query: &[f64], k: usize) -> Vec<usize> {
    if k == 0 {
        return Vec::new();
    }
    let mut dists: Vec<(usize, f64)> = points
        .iter()
        .enumerate()
        .map(|(i, p)| (i, dist2(p, query)))
        .collect();
    dists.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    dists.into_iter().take(k).map(|(i, _)| i).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_radius_and_knn_match_bruteforce() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![2.0, 2.0],
            vec![-1.0, 0.0],
        ];
        let tree = KDTree::build(points.clone()).expect("tree");
        let q = vec![0.2, 0.1];
        let mut r_tree = tree.radius_query(&q, 1.1);
        let mut r_brute = brute_radius(&points, &q, 1.1);
        r_tree.sort_unstable();
        r_brute.sort_unstable();
        assert_eq!(r_tree, r_brute);

        let k_tree = tree.knn_query(&q, 3);
        let k_brute = brute_knn(&points, &q, 3);
        assert_eq!(k_tree, k_brute);
    }
}
