use sprs::{CsMat, TriMat};

/// Returns tfidf.
pub fn get_tfidf(count_matrix: &CsMat<f64>) -> CsMat<f64> {
    let (n_documents, n_words) = count_matrix.shape();

    // TF: row-normalized counts.
    let mut tf_tri = TriMat::<f64>::new((n_documents, n_words));
    for (i, row) in count_matrix.outer_iterator().enumerate() {
        let row_sum: f64 = row.data().iter().sum();
        if row_sum > 0.0 {
            for (j, v) in row.iter() {
                tf_tri.add_triplet(i, j, v / row_sum);
            }
        }
    }
    let tf = tf_tri.to_csr::<usize>();

    // Document frequency per word.
    let mut freq = vec![0usize; n_words];
    for row in count_matrix.outer_iterator() {
        for &j in row.indices() {
            freq[j] += 1;
        }
    }

    // IDF = log(n_documents / freq) for freq > 0.
    let mut idf = vec![0.0f64; n_words];
    for j in 0..n_words {
        if freq[j] > 0 {
            idf[j] = ((n_documents as f64) / (freq[j] as f64)).ln();
        }
    }

    // TF-IDF: scale each term column by its IDF.
    let mut out_tri = TriMat::<f64>::new((n_documents, n_words));
    for (i, row) in tf.outer_iterator().enumerate() {
        for (j, v) in row.iter() {
            let value = v * idf[j];
            if value != 0.0 {
                out_tri.add_triplet(i, j, value);
            }
        }
    }
    out_tri.to_csr::<usize>()
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    fn dense_to_csr(dense: &ndarray::Array2<f64>) -> CsMat<f64> {
        let (r, c) = dense.dim();
        let mut tri = TriMat::<f64>::new((r, c));
        for i in 0..r {
            for j in 0..c {
                if dense[[i, j]] != 0.0 {
                    tri.add_triplet(i, j, dense[[i, j]]);
                }
            }
        }
        tri.to_csr::<usize>()
    }

    #[test]
    fn test_tfidf() {
        let count = dense_to_csr(&array![[0.0, 1.0, 2.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]);
        let tfidf = get_tfidf(&count);
        assert_eq!(count.shape(), tfidf.shape());
        assert_eq!(tfidf.nnz(), 2);
    }
}
