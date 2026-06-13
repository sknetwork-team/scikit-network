/// Returns indices of the top-`k` scores.
///
/// When `k >= scores.len()`, returns all indices. When `sort` is true, the
/// returned indices are ordered by descending score.
///
/// # Arguments
/// - `scores`: Node score vector.
/// - `k`: Number of top indices to return.
/// - `sort`: Whether to sort the returned indices by score.
pub fn top_k(scores: &[f64], k: usize, sort: bool) -> Vec<usize> {
    let n = scores.len();
    if n == 0 {
        return Vec::new();
    }

    if k >= n {
        if sort {
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_by(|&a, &b| {
                scores[b]
                    .partial_cmp(&scores[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            idx
        } else {
            (0..n).collect()
        }
    } else {
        let mut idx: Vec<usize> = (0..n).collect();
        idx.select_nth_unstable_by(k, |&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut top = idx[..k].to_vec();
        if sort {
            top.sort_by(|&a, &b| {
                scores[b]
                    .partial_cmp(&scores[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        top
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_k() {
        let scores: Vec<f64> = (0..10).map(|x| x as f64).collect();
        let index = top_k(&scores, 3, true);
        let set: std::collections::HashSet<usize> = index.into_iter().collect();
        assert_eq!(set, [7usize, 8, 9].into_iter().collect());

        let index = top_k(&scores, 10, true);
        assert_eq!(index.len(), 10);

        let index = top_k(&scores, 20, true);
        assert_eq!(index.len(), 10);

        let scores = vec![3.0, 1.0, 6.0, 2.0];
        let index = top_k(&scores, 2, true);
        let set: std::collections::HashSet<usize> = index.into_iter().collect();
        assert_eq!(set, [0usize, 2].into_iter().collect());

        let index = top_k(&scores, 2, true);
        assert_eq!(index, vec![2, 0]);
    }
}
