#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use ndarray::{Array1, Array2};
    use sprs::{CsMat, TriMat};

    use crate::classification::base::BaseClassifierError;
    use crate::classification::diffusion::{DiffusionClassifier, DiffusionClassifierError};
    use crate::classification::nn::{NNClassifier, NNClassifierError};
    use crate::classification::pagerank::{PageRankClassifier, PageRankClassifierError};
    use crate::classification::propagation::{Propagation, PropagationError};
    use crate::clustering::base::BaseClusteringError;
    use crate::clustering::leiden::{Leiden, LeidenError};
    use crate::clustering::louvain::{Louvain, LouvainError};
    use crate::clustering::propagation_clustering::{PropagationClustering, PropagationClusteringError};
    use crate::embedding::spring::{Spring, SpringError};
    use crate::gnn::base::{BaseGNN, GNNBaseError};
    use crate::gnn::gnn_classifier::GNNClassifier;
    use crate::hierarchy::louvain_hierarchy::HierarchyAlgoError;
    use crate::hierarchy::paris::Paris;
    use crate::ranking::hits::{HITS, HITSError};
    use crate::ranking::katz::{Katz, KatzError};
    use crate::ranking::pagerank::{PageRank, PageRankError};
    use crate::regression::diffusion::{Diffusion, RegressionError};
    use crate::utils::values::ValuesInput;

    fn tiny_graph() -> CsMat<f64> {
        let mut tri = TriMat::<f64>::new((4, 4));
        tri.add_triplet(0, 1, 1.0);
        tri.add_triplet(1, 0, 1.0);
        tri.add_triplet(1, 2, 1.0);
        tri.add_triplet(2, 1, 1.0);
        tri.add_triplet(2, 3, 1.0);
        tri.add_triplet(3, 2, 1.0);
        tri.to_csr::<usize>()
    }

    #[test]
    fn classification_propagation_contract_matrix() {
        let adjacency = tiny_graph();
        let mut algo = Propagation::new(5, Some("index"), true);

        // Pre-fit contract: inference endpoints must fail with NotFitted.
        assert_eq!(
            algo.predict(false),
            Err(PropagationError::Base(BaseClassifierError::NotFitted))
        );
        assert_eq!(
            algo.transform(false),
            Err(PropagationError::Base(BaseClassifierError::NotFitted))
        );

        // Post-fit contract: inference endpoints should succeed.
        let mut labels = HashMap::new();
        labels.insert(0usize, 0.0);
        labels.insert(1usize, 1.0);
        algo.fit(&adjacency, Some(ValuesInput::Map(labels)), None, None)
            .unwrap();
        assert_eq!(algo.predict(false).unwrap().len(), adjacency.rows());
        assert_eq!(algo.transform(false).unwrap().rows(), adjacency.rows());

        // Invalid parameter contract.
        let mut bad = Propagation::new(5, Some("bad-order"), true);
        let mut labels = HashMap::new();
        labels.insert(0usize, 0.0);
        labels.insert(1usize, 1.0);
        assert_eq!(
            bad.fit(&adjacency, Some(ValuesInput::Map(labels)), None, None),
            Err(PropagationError::UnknownNodeOrder)
        );
    }

    #[test]
    fn clustering_louvain_contract_matrix() {
        let adjacency = tiny_graph();
        let mut algo = Louvain::default();

        // Pre-fit contract: inference endpoints must fail with NotFitted.
        assert_eq!(
            algo.predict(false),
            Err(LouvainError::Base(BaseClusteringError::NotFitted))
        );
        assert_eq!(
            algo.transform(false),
            Err(LouvainError::Base(BaseClusteringError::NotFitted))
        );

        // Post-fit contract: inference endpoints should succeed.
        algo.fit(&adjacency, false).unwrap();
        assert_eq!(algo.predict(false).unwrap().len(), adjacency.rows());
        assert_eq!(algo.transform(false).unwrap().rows(), adjacency.rows());

        // Invalid parameter contract.
        let mut bad = Louvain::new(1.0, "bad-modularity", 1e-3, 1e-3, -1, false, true, true, true);
        assert_eq!(
            bad.fit(&adjacency, false),
            Err(LouvainError::UnknownModularity)
        );
    }

    #[test]
    fn embedding_spring_contract_matrix() {
        let adjacency = tiny_graph();
        let mut spring = Spring::default();

        // Pre-fit contract.
        assert_eq!(spring.transform(), Err(SpringError::NotFitted));

        // Post-fit contract.
        spring.fit(&adjacency, None, Some(5)).unwrap();
        let embedding = spring.transform().unwrap();
        assert_eq!(embedding.len(), adjacency.rows());

        // Invalid parameter contract.
        assert!(matches!(
            Spring::new(2, None, 50, 1e-4, -1.0, "bad-init"),
            Err(SpringError::UnknownPositionInit)
        ));
    }

    #[test]
    fn classification_nn_contract_matrix() {
        let adjacency = tiny_graph();
        let mut algo = NNClassifier::default();

        assert_eq!(
            algo.predict(false),
            Err(NNClassifierError::Base(BaseClassifierError::NotFitted))
        );
        assert_eq!(
            algo.transform(false),
            Err(NNClassifierError::Base(BaseClassifierError::NotFitted))
        );

        let mut labels = HashMap::new();
        labels.insert(0usize, 0.0);
        labels.insert(1usize, 1.0);
        algo.fit(&adjacency, Some(ValuesInput::Map(labels)), None, None)
            .unwrap();
        assert_eq!(algo.predict(false).unwrap().len(), adjacency.rows());
        assert_eq!(algo.transform(false).unwrap().rows(), adjacency.rows());

        let mut bad = NNClassifier::default();
        let no_seed = vec![-1.0; adjacency.rows()];
        assert_eq!(
            bad.fit(&adjacency, Some(ValuesInput::Vector(no_seed)), None, None),
            Err(NNClassifierError::NoSeedLabels)
        );
    }

    #[test]
    fn clustering_propagation_contract_matrix() {
        let adjacency = tiny_graph();
        let mut algo = PropagationClustering::default();

        assert_eq!(
            algo.predict(false),
            Err(PropagationClusteringError::Base(BaseClusteringError::NotFitted))
        );
        assert_eq!(
            algo.transform(false),
            Err(PropagationClusteringError::Base(BaseClusteringError::NotFitted))
        );

        algo.fit(&adjacency).unwrap();
        assert_eq!(algo.predict(false).unwrap().len(), adjacency.rows());
        assert_eq!(algo.transform(false).unwrap().rows(), adjacency.rows());

        let mut bad = PropagationClustering::new(5, "bad-order", true, true, true, true);
        assert_eq!(
            bad.fit(&adjacency),
            Err(PropagationClusteringError::UnknownNodeOrder)
        );
    }

    #[test]
    fn clustering_leiden_contract_matrix() {
        let adjacency = tiny_graph();
        let mut algo = Leiden::default();

        assert_eq!(
            algo.predict(false),
            Err(LeidenError::Louvain(LouvainError::Base(
                BaseClusteringError::NotFitted
            )))
        );
        assert_eq!(
            algo.transform(false),
            Err(LeidenError::Louvain(LouvainError::Base(
                BaseClusteringError::NotFitted
            )))
        );

        algo.fit(&adjacency, false).unwrap();
        assert_eq!(algo.predict(false).unwrap().len(), adjacency.rows());
        assert_eq!(algo.transform(false).unwrap().rows(), adjacency.rows());

        let mut bad = Leiden::new(1.0, "bad-modularity", 1e-3, 1e-3, -1, false, true, true, true);
        assert_eq!(
            bad.fit(&adjacency, false),
            Err(LeidenError::Louvain(LouvainError::UnknownModularity))
        );
    }

    #[test]
    fn hierarchy_paris_contract_matrix() {
        let adjacency = tiny_graph();
        let mut algo = Paris::default();

        assert_eq!(algo.predict(false), Err(HierarchyAlgoError::NotFitted));

        algo.fit(&adjacency, false).unwrap();
        assert_eq!(algo.predict(false).unwrap().len(), adjacency.rows() - 1);

        let mut bad = Paris::new("bad-weight", true);
        assert_eq!(
            bad.fit(&adjacency, false),
            Err(HierarchyAlgoError::UnknownWeights)
        );
    }

    #[test]
    fn gnn_classifier_contract_matrix() {
        let adjacency = tiny_graph();
        let features = Array2::<f64>::ones((adjacency.rows(), 4));
        let labels = Array1::from_vec((0..adjacency.rows()).map(|i| (i % 2) as i32).collect());
        let mut clf = GNNClassifier::new(vec![4, 8, 2], "cross_entropy", "adam", 1e-2, 2).unwrap();

        assert_eq!(clf.predict(&adjacency, &features), Err(GNNBaseError::NotFitted));
        assert_eq!(
            clf.transform(&adjacency, &features),
            Err(GNNBaseError::NotFitted)
        );

        clf.fit(&adjacency, &features, &labels).unwrap();
        assert_eq!(clf.predict(&adjacency, &features).unwrap().len(), adjacency.rows());
        assert_eq!(
            clf.transform(&adjacency, &features).unwrap().nrows(),
            adjacency.rows()
        );

        let mut bad = GNNClassifier::new(vec![4, 8, 3], "cross_entropy", "adam", 1e-2, 1).unwrap();
        assert_eq!(
            bad.fit(&adjacency, &features, &labels),
            Err(GNNBaseError::InvalidConfig)
        );
    }

    #[test]
    fn ranking_pagerank_contract_matrix() {
        let adjacency = tiny_graph();
        let mut algo = PageRank::default();

        // Non-fallible predict contract for ranking estimators in current API.
        assert!(algo.predict(false).is_empty());

        algo.fit(&adjacency, None, None, None, false).unwrap();
        assert_eq!(algo.predict(false).len(), adjacency.rows());

        assert!(matches!(
            PageRank::new(1.2, 10, 1e-6),
            Err(PageRankError::InvalidDampingFactor)
        ));
    }

    #[test]
    fn ranking_katz_contract_matrix() {
        let adjacency = tiny_graph();
        let mut algo = Katz::default();

        // Non-fallible predict contract for ranking estimators in current API.
        assert!(algo.predict(false).is_empty());

        algo.fit(&adjacency).unwrap();
        assert_eq!(algo.predict(false).len(), adjacency.rows());

        let bad = CsMat::<f64>::zero((0, 0));
        assert_eq!(algo.fit(&bad), Err(KatzError::InvalidInput));
    }

    #[test]
    fn ranking_hits_contract_matrix() {
        let adjacency = tiny_graph();
        let mut algo = HITS::default();

        // Non-fallible predict contract for ranking estimators in current API.
        assert!(algo.predict(false).is_empty());
        assert!(algo.predict(true).is_empty());

        algo.fit(&adjacency).unwrap();
        assert_eq!(algo.predict(false).len(), adjacency.rows());
        assert_eq!(algo.predict(true).len(), adjacency.cols());

        let bad = CsMat::<f64>::zero((0, 0));
        assert_eq!(algo.fit(&bad), Err(HITSError::InvalidInput));
    }

    #[test]
    fn classification_pagerank_contract_matrix() {
        let adjacency = tiny_graph();
        let mut algo = PageRankClassifier::default();

        assert_eq!(
            algo.predict(false),
            Err(PageRankClassifierError::Base(BaseClassifierError::NotFitted))
        );
        assert_eq!(
            algo.transform(false),
            Err(PageRankClassifierError::Base(BaseClassifierError::NotFitted))
        );

        let mut labels = HashMap::new();
        labels.insert(0usize, 0.0);
        labels.insert(1usize, 1.0);
        algo.fit(&adjacency, Some(ValuesInput::Map(labels)), None, None)
            .unwrap();
        assert_eq!(algo.predict(false).unwrap().len(), adjacency.rows());
        assert_eq!(algo.transform(false).unwrap().rows(), adjacency.rows());

        let bad = PageRankClassifier::new(2.0, 10, 1e-6);
        assert!(matches!(
            bad,
            Err(PageRankClassifierError::PageRank(
                PageRankError::InvalidDampingFactor
            ))
        ));
    }

    #[test]
    fn classification_diffusion_contract_matrix() {
        let adjacency = tiny_graph();
        let mut algo = DiffusionClassifier::default();

        assert_eq!(
            algo.predict(false),
            Err(DiffusionClassifierError::Base(BaseClassifierError::NotFitted))
        );
        assert_eq!(
            algo.transform(false),
            Err(DiffusionClassifierError::Base(BaseClassifierError::NotFitted))
        );

        let mut labels = HashMap::new();
        labels.insert(0usize, 0.0);
        labels.insert(1usize, 1.0);
        algo.fit(&adjacency, Some(ValuesInput::Map(labels)), None, None, false)
            .unwrap();
        assert_eq!(algo.predict(false).unwrap().len(), adjacency.rows());
        assert_eq!(algo.transform(false).unwrap().rows(), adjacency.rows());

        assert!(matches!(
            DiffusionClassifier::new(0, true, 1.0),
            Err(DiffusionClassifierError::InvalidNIter)
        ));
    }

    #[test]
    fn regression_diffusion_contract_matrix() {
        let adjacency = tiny_graph();
        let mut algo = Diffusion::default();

        assert_eq!(algo.predict(false), Err(RegressionError::NotFitted));

        algo.fit(&adjacency, None, None, None, None, false).unwrap();
        assert_eq!(algo.predict(false).unwrap().len(), adjacency.rows());

        assert!(matches!(
            Diffusion::new(0, 0.5),
            Err(RegressionError::InvalidNIter)
        ));
        assert!(matches!(
            Diffusion::new(2, 1.5),
            Err(RegressionError::InvalidDampingFactor)
        ));
    }
}
