use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
/// ValuesInput enum.
pub enum ValuesInput {
    /// Indicates vector.
    Vector(Vec<f64>),
    /// Indicates map.
    Map(HashMap<usize, f64>),
    /// Use the default value for every node.
    None,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Errors raised by values error operations.
pub enum ValuesError {
    /// Indicates invalid shape.
    InvalidShape,
    /// Indicates dimension mismatch.
    DimensionMismatch,
    /// Indicates index out of bounds.
    IndexOutOfBounds(usize),
    /// Indicates no positive value.
    NoPositiveValue,
}

/// Returns values.
pub fn get_values(
    shape: &[usize],
    values: ValuesInput,
    default_value: f64,
) -> Result<Vec<f64>, ValuesError> {
    let Some(&n) = shape.first() else {
        return Err(ValuesError::InvalidShape);
    };

    match values {
        ValuesInput::Vector(v) => {
            if v.len() != n {
                return Err(ValuesError::DimensionMismatch);
            }
            Ok(v)
        }
        ValuesInput::Map(m) => {
            let mut out = vec![default_value; n];
            for (k, v) in m {
                if k >= n {
                    return Err(ValuesError::IndexOutOfBounds(k));
                }
                out[k] = v;
            }
            Ok(out)
        }
        ValuesInput::None => Ok(vec![default_value; n]),
    }
}

/// Computes stack values.
pub fn stack_values(
    shape: (usize, usize),
    values_row: Option<ValuesInput>,
    values_col: Option<ValuesInput>,
    default_value: f64,
) -> Result<Vec<f64>, ValuesError> {
    let (n_row, n_col) = shape;

    let row_input = match values_row {
        Some(v) => v,
        None => {
            if values_col.is_none() {
                ValuesInput::Vector(vec![1.0; n_row])
            } else {
                ValuesInput::Vector(vec![default_value; n_row])
            }
        }
    };

    let col_input = match values_col {
        Some(v) => v,
        None => ValuesInput::Vector(vec![default_value; n_col]),
    };

    let mut row_values = get_values(&[n_row], row_input, default_value)?;
    let col_values = get_values(&[n_col], col_input, default_value)?;
    row_values.extend(col_values);
    Ok(row_values)
}

/// Computes values to prob.
///
/// # Errors
///
/// Returns [`ValuesError`] on failure.
pub fn values_to_prob(n: usize, values: Option<ValuesInput>) -> Result<Vec<f64>, ValuesError> {
    if n == 0 {
        return Err(ValuesError::InvalidShape);
    }

    let Some(input_values) = values else {
        let p = 1.0 / (n as f64);
        return Ok(vec![p; n]);
    };

    let source = get_values(&[n], input_values, -1.0)?;
    let mut probs = vec![0.0; n];
    let mut sum = 0.0;

    for (i, value) in source.into_iter().enumerate() {
        if value > 0.0 {
            probs[i] = value;
            sum += value;
        }
    }

    if sum <= 0.0 {
        return Err(ValuesError::NoPositiveValue);
    }

    for prob in &mut probs {
        *prob /= sum;
    }
    Ok(probs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() <= tol, "left={x}, right={y}");
        }
    }

    #[test]
    fn test_get_values_vector_and_map() {
        let n = 10;
        let mut labels = vec![-1.0; n];
        labels[0] = 0.0;
        labels[1] = 1.0;

        let labels_from_vec = get_values(&[n], ValuesInput::Vector(labels.clone()), -1.0).unwrap();

        let mut map = HashMap::new();
        map.insert(0, 0.0);
        map.insert(1, 1.0);
        let labels_from_map = get_values(&[n], ValuesInput::Map(map), -1.0).unwrap();

        assert_close(&labels_from_vec, &labels_from_map, 1e-12);
    }

    #[test]
    fn test_get_values_errors() {
        let out = get_values(&[5], ValuesInput::Vector(vec![0.0; 10]), -1.0);
        assert_eq!(out, Err(ValuesError::DimensionMismatch));

        let out = get_values(&[], ValuesInput::None, -1.0);
        assert_eq!(out, Err(ValuesError::InvalidShape));
    }

    #[test]
    fn test_values_to_prob() {
        let n = 4;
        let values_array = vec![0.0, 1.0, -1.0, 0.0];
        let probs1 = values_to_prob(n, Some(ValuesInput::Vector(values_array))).unwrap();

        let mut values_map = HashMap::new();
        values_map.insert(0, 0.0);
        values_map.insert(1, 1.0);
        values_map.insert(3, 0.0);
        let probs2 = values_to_prob(n, Some(ValuesInput::Map(values_map))).unwrap();
        assert_close(&probs1, &probs2, 1e-12);

        let bad_input = vec![0.0, 0.0, -1.0, 0.0];
        let out = values_to_prob(n, Some(ValuesInput::Vector(bad_input)));
        assert_eq!(out, Err(ValuesError::NoPositiveValue));
    }

    #[test]
    fn test_stack_values() {
        let shape = (4, 3);
        let values_row_array = vec![0.0, 1.0, -1.0, 0.0];
        let values_col_array = vec![0.0, 1.0, -1.0];

        let values1 = stack_values(
            shape,
            Some(ValuesInput::Vector(values_row_array.clone())),
            Some(ValuesInput::Vector(values_col_array.clone())),
            -1.0,
        )
        .unwrap();

        let mut values_row_map = HashMap::new();
        values_row_map.insert(0, 0.0);
        values_row_map.insert(1, 1.0);
        values_row_map.insert(3, 0.0);

        let mut values_col_map = HashMap::new();
        values_col_map.insert(0, 0.0);
        values_col_map.insert(1, 1.0);

        let values2 = stack_values(
            shape,
            Some(ValuesInput::Map(values_row_map.clone())),
            Some(ValuesInput::Map(values_col_map.clone())),
            -1.0,
        )
        .unwrap();
        assert_close(&values1, &values2, 1e-12);

        let values3 = stack_values(
            shape,
            Some(ValuesInput::Vector(values_row_array.clone())),
            Some(ValuesInput::Map(values_col_map)),
            -1.0,
        )
        .unwrap();
        assert_close(&values2, &values3, 1e-12);

        let values4 = stack_values(
            shape,
            Some(ValuesInput::Map(values_row_map)),
            Some(ValuesInput::Vector(values_col_array)),
            -1.0,
        )
        .unwrap();
        assert_close(&values3, &values4, 1e-12);
    }
}
